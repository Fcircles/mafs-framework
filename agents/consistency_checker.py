"""一致性自检Agent -- 检验计算口径一致性与逻辑一致性"""

from __future__ import annotations

import json as _json_mod
import re
import logging
import time

from agents import PipelineState
from utils.llm_client import get_client_for_provider, get_model_for_provider, chat_completion

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 3


# ---------------------------------------------------------------------------
# A. 规则检查
# ---------------------------------------------------------------------------

def _check_data_completeness(indicators: dict) -> list[dict]:
    """检查关键指标是否全部计算成功。

    缺失指标类别属于数据提取阶段的问题，重试解读生成无法修复，
    因此标记为 medium 而非 high。
    """
    issues: list[dict] = []
    required_categories = ["杜邦分析", "Z-Score", "偿债能力", "盈利能力", "营运能力"]

    for cat in required_categories:
        if cat not in indicators:
            issues.append({
                "type": "data_completeness",
                "severity": "medium",
                "detail": f"缺失指标类别: {cat}",
            })

    missing = indicators.get("计算缺失项", [])
    for m in missing:
        issues.append({
            "type": "data_completeness",
            "severity": "medium",
            "detail": f"计算缺失: {m}",
        })

    return issues


def _check_direction_consistency(indicators: dict, interpretation: str) -> list[dict]:
    """检查解读结论的方向是否与数据一致。"""
    issues: list[dict] = []
    if not interpretation:
        return issues

    dupont = indicators.get("杜邦分析", {})
    roe = dupont.get("ROE")
    if roe is not None:
        if roe > 0 and ("亏损" in interpretation and "盈利" not in interpretation):
            issues.append({
                "type": "direction_mismatch",
                "severity": "high",
                "detail": f"ROE={roe:.4f}>0，但解读提及亏损",
            })
        if roe < 0 and "盈利良好" in interpretation:
            issues.append({
                "type": "direction_mismatch",
                "severity": "high",
                "detail": f"ROE={roe:.4f}<0，但解读称盈利良好",
            })

    zscore = indicators.get("Z-Score", {})
    zone = zscore.get("zone")
    z_val = zscore.get("Z''-Score") or zscore.get("Z-Score")
    has_caveat = bool(zscore.get("model_caveat"))
    if zone and z_val is not None:
        if zone == "危险区" and "危险区" not in interpretation:
            severity = "medium" if has_caveat else "high"
            issues.append({
                "type": "direction_mismatch",
                "severity": severity,
                "detail": f"Z-Score={z_val:.2f}处于危险区，但解读未提及危险区"
                          + ("（模型已标注局限性）" if has_caveat else ""),
            })
        if zone == "安全区" and "高风险" in interpretation:
            issues.append({
                "type": "direction_mismatch",
                "severity": "medium",
                "detail": f"Z-Score={z_val:.2f}处于安全区，但解读称高风险",
            })

    solvency = indicators.get("偿债能力", {})
    debt_ratio = solvency.get("资产负债率")
    if debt_ratio is not None:
        if debt_ratio > 0.7 and "负债水平较低" in interpretation:
            issues.append({
                "type": "direction_mismatch",
                "severity": "medium",
                "detail": f"资产负债率={debt_ratio:.2%}偏高，但解读称负债水平较低",
            })

    return issues


def _check_value_citation(indicators: dict, interpretation: str) -> list[dict]:
    """检查解读中引用的数值是否与计算结果大致匹配。"""
    issues: list[dict] = []
    if not interpretation:
        return issues

    numbers_in_text = re.findall(r"(\d+\.?\d*)\s*%", interpretation)

    all_values: dict[str, float] = {}
    for cat, data in indicators.items():
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)):
                all_values[k] = v

    for num_str in numbers_in_text:
        try:
            cited_pct = float(num_str) / 100.0
        except ValueError:
            continue

        for ind_name, ind_val in all_values.items():
            if abs(ind_val) < 0.0001:
                continue
            if abs(cited_pct - ind_val) / abs(ind_val) <= 0.1:
                break
        else:
            issues.append({
                "type": "value_citation_mismatch",
                "severity": "low",
                "detail": f"解读中引用了 {num_str}%，未找到匹配的计算指标值",
            })

    return issues


def _rule_based_checks(indicators: dict, interpretation: str) -> list[dict]:
    """执行所有规则检查。"""
    issues: list[dict] = []
    issues.extend(_check_data_completeness(indicators))
    issues.extend(_check_direction_consistency(indicators, interpretation))
    issues.extend(_check_value_citation(indicators, interpretation))
    return issues


# ---------------------------------------------------------------------------
# B. LLM 语义检查
# ---------------------------------------------------------------------------

SEMANTIC_CHECK_PROMPT = """\
你是一名财务审计专家。请检查以下财务分析报告与原始计算指标之间是否存在逻辑不一致。

## 检查要求
1. 报告中的结论方向是否与数据指标的变动方向一致
2. 报告中引用的比率数值是否与指标数据匹配
3. Z-Score/Z''-Score 区间判断是否正确；如指标中有 model_caveat 字段，报告引用该局限性说明不算矛盾
4. 是否存在数据未支持的主观推断

## 已知的正常差异（以下情况不算问题）
- 杜邦分析的ROE使用平均权益计算，盈利能力的ROE使用期末权益计算，两者数值不同是正常的计算口径差异
- 权益乘数使用平均总资产/平均权益，与用期末资产负债率换算的值不同是正常的
- 报告中引用的绝对金额如来自"已验证合并报表数据"部分则为可靠数据

## 财务指标数据
{indicators_text}

## 财务分析报告
{interpretation}

## 输出格式
只报告真正的逻辑错误或数据矛盾，忽略上述已知的正常差异。
请以JSON数组格式返回发现的问题（如无问题则返回空数组 []）：
[
  {{"issue": "问题描述", "severity": "high/medium/low"}},
  ...
]
仅返回JSON数组，不要添加其他文字。
"""


def _llm_semantic_check(
    indicators: dict,
    interpretation: str,
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> list[dict]:
    """使用 LLM 进行语义级一致性检查。"""
    if not interpretation:
        return []

    ind_lines: list[str] = []
    for cat, data in indicators.items():
        if cat == "计算缺失项" or not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, float):
                ind_lines.append(f"{cat}/{k}: {v:.4f}")
            else:
                ind_lines.append(f"{cat}/{k}: {v}")

    prompt = SEMANTIC_CHECK_PROMPT.format(
        indicators_text="\n".join(ind_lines),
        interpretation=interpretation[:3000],
    )

    client = get_client_for_provider(llm_provider)
    _model = llm_model or get_model_for_provider(llm_provider)
    issues: list[dict] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = chat_completion(
                client, _model,
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            bracket_start = raw.find("[")
            bracket_end = raw.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                raw = raw[bracket_start:bracket_end + 1]

            parsed = _json_mod.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    issues.append({
                        "type": "semantic_check",
                        "severity": item.get("severity", "medium"),
                        "detail": item.get("issue", str(item)),
                    })
            break
        except ValueError:
            logger.warning("LLM 语义检查输出非合法 JSON (第%d次)", attempt)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as exc:
            logger.warning("LLM 语义检查失败 (第%d次): %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return issues


# ---------------------------------------------------------------------------
# C. 跨年一致性检查（预留接口，供 orchestrator 外部调用）
# ---------------------------------------------------------------------------

def check_cross_year_consistency(all_year_results: list[dict]) -> dict:
    """检查同一公司多年指标计算的口径一致性。

    参数:
        all_year_results: 多年运行结果列表，每项包含 year 和 indicators 键。

    返回:
        {"issues": [...], "summary": "..."}
    """
    issues: list[dict] = []

    if len(all_year_results) < 2:
        return {"issues": [], "summary": "数据不足两年，无法进行跨年比较"}

    sorted_results = sorted(all_year_results, key=lambda r: r.get("year", 0))

    for i in range(1, len(sorted_results)):
        prev = sorted_results[i - 1]
        curr = sorted_results[i]
        prev_inds = prev.get("indicators", {})
        curr_inds = curr.get("indicators", {})
        prev_year = prev.get("year", "?")
        curr_year = curr.get("year", "?")

        prev_cats = {k for k in prev_inds if k != "计算缺失项" and isinstance(prev_inds[k], dict)}
        curr_cats = {k for k in curr_inds if k != "计算缺失项" and isinstance(curr_inds[k], dict)}

        only_prev = prev_cats - curr_cats
        only_curr = curr_cats - prev_cats
        if only_prev or only_curr:
            issues.append({
                "type": "category_mismatch",
                "severity": "medium",
                "detail": (f"{prev_year}->{curr_year}: "
                           f"前期独有类别{only_prev}, 后期独有类别{only_curr}"),
            })

        for cat in prev_cats & curr_cats:
            prev_keys = set(prev_inds[cat].keys())
            curr_keys = set(curr_inds[cat].keys())
            diff = prev_keys.symmetric_difference(curr_keys)
            if diff:
                issues.append({
                    "type": "indicator_mismatch",
                    "severity": "low",
                    "detail": (f"{prev_year}->{curr_year} [{cat}]: "
                               f"指标口径差异 {diff}"),
                })

    summary = f"检查了{len(sorted_results)}年数据，发现{len(issues)}个口径一致性问题"
    return {"issues": issues, "summary": summary}


# ---------------------------------------------------------------------------
# Agent 节点函数
# ---------------------------------------------------------------------------

MAX_CONSISTENCY_RETRIES = 2


def consistency_checker_node(state: PipelineState) -> dict:
    """一致性自检节点：规则检查 + LLM 语义检查。"""

    indicators = state.get("indicators", {})
    interpretation = state.get("interpretation", "")
    errors: list[str] = list(state.get("errors", []))
    retry_count = state.get("retry_count", 0)

    if not interpretation:
        logger.warning("解读内容为空，跳过一致性检查（标记为未通过）")
        consistency_report = {
            "rule_checks": [{"type": "empty_interpretation",
                             "severity": "high",
                             "detail": "解读生成返回空内容"}],
            "semantic_checks": [],
            "summary": {"total_issues": 1, "high": 1, "medium": 0,
                         "low": 0, "passed": False},
        }
        return {
            "consistency_report": consistency_report,
            "consistency_passed": False,
            "retry_count": retry_count + 1,
            "errors": errors,
        }

    rule_issues = _rule_based_checks(indicators, interpretation)
    semantic_issues: list[dict] = []

    _provider = state.get("llm_provider", "tengri")
    _model = state.get("llm_model", "")

    try:
        semantic_issues = _llm_semantic_check(
            indicators, interpretation,
            llm_provider=_provider, llm_model=_model,
        )
    except Exception as exc:
        msg = f"LLM 语义检查异常: {exc}"
        logger.error(msg)
        errors.append(msg)

    all_issues = rule_issues + semantic_issues
    high = sum(1 for i in all_issues if i.get("severity") == "high")
    medium = sum(1 for i in all_issues if i.get("severity") == "medium")
    low = sum(1 for i in all_issues if i.get("severity") == "low")

    passed = high == 0

    consistency_report = {
        "rule_checks": rule_issues,
        "semantic_checks": semantic_issues,
        "summary": {
            "total_issues": len(all_issues),
            "high": high,
            "medium": medium,
            "low": low,
            "passed": passed,
        },
    }

    logger.info("一致性自检完成: %d 个问题 (高%d/中%d/低%d), 通过=%s, 重试=%d/%d",
                 len(all_issues), high, medium, low,
                 passed, retry_count, MAX_CONSISTENCY_RETRIES)

    return {
        "consistency_report": consistency_report,
        "consistency_passed": passed,
        "retry_count": retry_count + 1,
        "errors": errors,
    }
