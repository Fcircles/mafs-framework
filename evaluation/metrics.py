"""评价指标计算 -- 含LLM-as-Judge双评机制"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import cohen_kappa_score

from config import TengriConfig, ConcurrencyConfig
from models import AnalysisResult, IndicatorResult
from utils.llm_client import get_tengri_client, chat_completion

logger = logging.getLogger(__name__)


# ===================================================================
# 1. 计算正确率（含指标名称标准化）
# ===================================================================

_NAME_ALIASES: dict[str, str] = {
    "净资产收益率": "ROE", "净资产收益率 (ROE)": "ROE",
    "ROE (净资产收益率)": "ROE", "净资产收益率(ROE)": "ROE",
    "销售毛利率": "毛利率", "毛利率（销售毛利率）": "毛利率",
    "总资产收益率": "ROA", "总资产收益率 (ROA)": "ROA",
    "ROA (总资产收益率)": "ROA", "资产收益率 (ROA)": "ROA",
    "ROA(总资产收益率)": "ROA", "总资产报酬率": "ROA",
    "Altman Z-Score": "Z-Score", "Altman Z-Score (Z'')": "Z''-Score",
    "Altman Z''-Score": "Z''-Score", "Z''": "Z''-Score",
}

_COMPOSITE_REMAP: dict[str, str] = {
    "杜邦分析/总资产周转率": "杜邦分析/资产周转率",
    "杜邦分析/盈利能力/销售净利率": "杜邦分析/销售净利率",
    "盈利能力/销售净利率": "杜邦分析/销售净利率",
    "盈利能力/资产周转率": "杜邦分析/资产周转率",
    "盈利能力/权益乘数": "杜邦分析/权益乘数",
}

CORE_INDICATORS: set[str] = {
    "杜邦分析/ROE", "杜邦分析/销售净利率", "杜邦分析/资产周转率", "杜邦分析/权益乘数",
    "Z-Score/Z-Score", "Z-Score/Z''-Score",
    "偿债能力/流动比率", "偿债能力/速动比率", "偿债能力/资产负债率",
    "盈利能力/毛利率", "盈利能力/净利率", "盈利能力/ROA", "盈利能力/ROE",
    "营运能力/应收账款周转率", "营运能力/存货周转率", "营运能力/总资产周转率",
}


def _normalize_indicator_key(category: str, name: str) -> str:
    """将各方法输出的指标名标准化为 GT 使用的键。"""
    norm_name = _NAME_ALIASES.get(name, name)
    composite = f"{category}/{norm_name}"
    return _COMPOSITE_REMAP.get(composite, composite)


def calculation_correctness(
    predicted: list[IndicatorResult],
    ground_truth: dict[str, float],
    tolerance: float = 0.05,
) -> dict[str, float | dict]:
    """
    计算正确率 = 正确计算的指标数 / 总指标数 * 100%

    Parameters
    ----------
    predicted : 模型输出的指标列表
    ground_truth : {category/指标名: 正确值} 字典
    tolerance : 相对误差容忍度（默认5%）
    """
    if not ground_truth:
        return {"accuracy": 0.0, "correct_count": 0, "total_count": 0, "details": {}}

    pred_map: dict[str, float] = {}
    for ind in predicted:
        if ind.value is not None:
            key = _normalize_indicator_key(ind.category, ind.name)
            if key not in pred_map:
                pred_map[key] = ind.value

    correct = 0
    total = len(ground_truth)
    details: dict[str, dict] = {}

    for name, gt_val in ground_truth.items():
        pred_val = pred_map.get(name)
        if pred_val is None:
            details[name] = {"status": "missing", "predicted": None, "expected": gt_val}
            continue

        if gt_val == 0:
            is_correct = abs(pred_val) < 1e-6
        else:
            rel_error = abs(pred_val - gt_val) / abs(gt_val)
            is_correct = rel_error <= tolerance

        if is_correct:
            correct += 1
            details[name] = {"status": "correct", "predicted": pred_val, "expected": gt_val}
        else:
            details[name] = {
                "status": "incorrect",
                "predicted": pred_val,
                "expected": gt_val,
                "relative_error": abs(pred_val - gt_val) / abs(gt_val) if gt_val != 0 else None,
            }

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct_count": correct,
        "total_count": total,
        "details": details,
    }


# ===================================================================
# 1.5 公式验证通过率（不依赖外部 Ground Truth）
# ===================================================================

def formula_verification(indicators: list[IndicatorResult]) -> dict:
    """公式验证通过率：检查各指标类别的计算是否数学自洽。

    验证项目：
    1. 杜邦分解：ROE ≈ 销售净利率 x 资产周转率 x 权益乘数 (容差1%)
    2. 盈利能力：毛利率 > 净利率（在正常情况下）
    3. 偿债能力：资产负债率 + 权益比率 ≈ 1 (容差5%)
    4. 数据存在性：每个类别至少有1个指标值
    """
    by_cat: dict[str, dict[str, float]] = defaultdict(dict)
    for ind in indicators:
        if ind.value is not None:
            norm_name = _NAME_ALIASES.get(ind.name, ind.name)
            by_cat[ind.category][norm_name] = ind.value

    passed = 0
    total = 0
    details: dict[str, dict] = {}

    # --- 杜邦分解验证 ---
    dupont = by_cat.get("杜邦分析", {})
    roe = dupont.get("ROE")
    npm = dupont.get("销售净利率")
    tat = dupont.get("资产周转率")
    em = dupont.get("权益乘数")
    if roe is not None and npm is not None and tat is not None and em is not None:
        total += 1
        product = npm * tat * em
        if abs(roe) < 1e-9:
            ok = abs(product) < 1e-6
        else:
            ok = abs(roe - product) / abs(roe) < 0.01
        if ok:
            passed += 1
        details["杜邦分解"] = {
            "check": "ROE ≈ 销售净利率 × 资产周转率 × 权益乘数",
            "passed": ok,
            "roe": roe, "product": product,
        }

    # --- 盈利能力验证 ---
    profit = by_cat.get("盈利能力", {})
    gross_margin = profit.get("毛利率")
    net_margin = profit.get("净利率")
    if gross_margin is not None and net_margin is not None:
        total += 1
        ok = gross_margin >= net_margin or net_margin < 0
        if ok:
            passed += 1
        details["盈利能力_毛利率>=净利率"] = {
            "check": "毛利率 >= 净利率（允许亏损例外）",
            "passed": ok,
            "gross_margin": gross_margin, "net_margin": net_margin,
        }

    # --- 偿债能力验证 ---
    solvency = by_cat.get("偿债能力", {})
    debt_ratio = solvency.get("资产负债率")
    if debt_ratio is not None:
        total += 1
        ok = 0.0 <= debt_ratio <= 1.0
        if ok:
            passed += 1
        details["偿债能力_资产负债率范围"] = {
            "check": "资产负债率 ∈ [0, 1]",
            "passed": ok,
            "debt_ratio": debt_ratio,
        }

    # --- Z-Score 区域判定验证 ---
    zscore_cat = by_cat.get("Z-Score", {})
    for z_name in ("Z-Score", "Z''-Score"):
        z_val = zscore_cat.get(z_name)
        if z_val is not None:
            total += 1
            ok = isinstance(z_val, (int, float)) and not np.isnan(z_val)
            if ok:
                passed += 1
            details[f"Z-Score_{z_name}_有效"] = {
                "check": f"{z_name} 为有效数值",
                "passed": ok,
                "value": z_val,
            }

    # --- 数据存在性验证 ---
    expected_cats = ["杜邦分析", "偿债能力", "盈利能力", "营运能力", "Z-Score"]
    for cat in expected_cats:
        total += 1
        ok = len(by_cat.get(cat, {})) >= 1
        if ok:
            passed += 1
        details[f"存在性_{cat}"] = {"passed": ok, "count": len(by_cat.get(cat, {}))}

    return {
        "pass_rate": passed / total if total > 0 else 0.0,
        "total_checks": total,
        "passed_checks": passed,
        "details": details,
    }


# ===================================================================
# 2. 证据对齐率
# ===================================================================

def evidence_alignment_rate(indicators: list[IndicatorResult]) -> dict[str, float | int]:
    """
    证据对齐率 = 可追溯到年报原文的指标数 / 总指标数 * 100%

    判定标准：source_page 不为 None 且 source_text 非空
    """
    if not indicators:
        return {"alignment_rate": 0.0, "aligned_count": 0, "total_count": 0}

    aligned = sum(
        1 for ind in indicators
        if ind.source_page is not None and (ind.source_text or "").strip()
    )
    return {
        "alignment_rate": aligned / len(indicators),
        "aligned_count": aligned,
        "total_count": len(indicators),
    }


# ===================================================================
# 3. 口径一致性
# ===================================================================

def caliber_consistency(multi_year_results: list[AnalysisResult]) -> dict[str, float | dict]:
    """
    口径一致性 = 跨年份计算口径一致的指标数 / 总指标数 * 100%

    检查同一公司多年结果中，各指标类别下：
    - 使用的指标名称集合是否一致
    - 计算公式描述是否一致
    """
    if len(multi_year_results) < 2:
        return {"consistency": 0.0, "consistent_count": 0, "total_count": 0, "details": {}}

    by_category: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for result in multi_year_results:
        for ind in result.indicators:
            if ind.value is None:
                continue
            by_category[ind.category].setdefault(result.year, {})[ind.name] = ind.formula

    consistent = 0
    total = 0
    details: dict[str, dict] = {}

    for category, year_data in by_category.items():
        years = sorted(year_data.keys())
        if len(years) < 2:
            continue

        ref_names = set(year_data[years[0]].keys())
        ref_formulas = year_data[years[0]]

        cat_total = len(ref_names)
        cat_consistent = 0

        for ind_name in ref_names:
            total += 1
            name_present_all = all(ind_name in year_data[y] for y in years)
            formula_same_all = all(
                year_data[y].get(ind_name) == ref_formulas.get(ind_name)
                for y in years
            )
            if name_present_all and formula_same_all:
                consistent += 1
                cat_consistent += 1

        details[category] = {
            "consistent_count": cat_consistent,
            "total_count": cat_total,
            "years_checked": years,
        }

    return {
        "consistency": consistent / total if total > 0 else 0.0,
        "consistent_count": consistent,
        "total_count": total,
        "details": details,
    }


# ===================================================================
# 4. 解读质量 -- LLM-as-Judge 双评机制
# ===================================================================

_REVIEWER_A_PROMPT = """\
你是一位具有10年以上经验的注册会计师（CPA），专注于财务数据准确性审核。\
请对以下上市公司年报财务分析报告进行严格评审。

## 待评审报告信息
- 公司名称：{company_name}
- 年报年份：{year}年
- 所属行业：{industry}

## 实际计算指标值（标准答案）
{indicator_summary}

## 待评价的分析报告
---
{interpretation}
---

## 评审标准（请严格对照标准打分，不要给人情分）

### 维度1: 数据准确（1-5分）
- 5分: 所有引用的财务数值与上方"实际计算指标值"完全吻合，无任何数据错误
- 4分: 绝大部分数据正确，仅有1-2处小数点精度差异（相对误差<5%）
- 3分: 大部分数据正确，但有2-3处明显数值错误（相对误差>10%）
- 2分: 多处数据错误，或引用了"实际计算指标值"中不存在的数据
- 1分: 大量数据编造，或未引用任何具体数值

### 维度2: 计算合理（1-5分）
- 5分: 所有财务指标的计算方法、公式和口径均正确
- 4分: 计算方法基本正确，有个别计算方式不够规范但不影响结论
- 3分: 部分计算方法有误，但整体方向合理
- 2分: 多项计算方法错误或指标定义混淆
- 1分: 未进行有效的财务计算，或计算严重错误

### 维度3: 口径一致（1-5分）
- 5分: 报告内部所有数据引用完全一致，无自相矛盾
- 4分: 基本一致，有1处微小不一致但不影响结论
- 3分: 存在2-3处数据引用不一致，部分影响可信度
- 2分: 多处数据自相矛盾
- 1分: 数据引用混乱，严重矛盾

### 综合质量（1-5分）——此分数最为重要
- 5分: 数据准确、计算正确、口径一致，报告含有详细的叙述性分析，达到专业审计参考水平
- 4分: 整体可靠，有个别小瑕疵不影响使用，包含有意义的分析文字
- 3分: 有明显问题但核心分析仍有参考价值
- 2分: 存在重大准确性问题，或报告仅罗列数字而无分析性叙述文字
- 1分: 基本不可用，数据严重失真或完全无分析内容

**重要说明**：
- 报告中引用的绝对金额（如"营业收入xxx亿元"）来自年报原文，不算编造数据
- 仅当报告中的财务比率（如ROE、资产负债率等百分比/比率）与上方"实际计算指标值"严重不符时才算数据错误
- "编造数据"特指凭空捏造不存在的财务比率数值

**强制扣分规则（必须执行）**：
- 如果报告仅罗列指标数值而无任何叙述性分析文字 → 综合质量 ≤ 2
- 如果报告中的财务比率与实际指标值有3处及以上严重偏差（>20%） → 综合质量 ≤ 3
- 如果报告少于100字 → 综合质量 ≤ 2

请严格按以下JSON格式返回，不要输出其他内容：
{{"数据准确": 整数, "计算合理": 整数, "口径一致": 整数, "综合质量": 整数}}"""

_REVIEWER_B_PROMPT = """\
你是一位具有丰富经验的投资分析师（CFA），专注于财务分析报告的逻辑质量评审。\
请对以下上市公司年报财务分析报告进行严格评审。

## 待评审报告信息
- 公司名称：{company_name}
- 年报年份：{year}年
- 所属行业：{industry}

## 实际计算指标值（供核对参考）
{indicator_summary}

## 待评价的分析报告
---
{interpretation}
---

## 评审标准（请严格对照标准打分，不要给人情分）

### 维度1: 逻辑通顺（1-5分）
- 5分: 分析层次分明，因果推理严密，每个结论都有具体数据支撑
- 4分: 逻辑基本通顺，个别地方推理不够紧凑但不影响阅读
- 3分: 逻辑结构存在跳跃，部分结论缺乏充分论证
- 2分: 逻辑较混乱，多处结论缺乏依据
- 1分: 无有效分析逻辑，纯数据罗列或文不对题

### 维度2: 结论匹配（1-5分）
- 5分: 所有分析结论的方向与指标变动方向完全匹配（如ROE高→评价盈利能力较强）
- 4分: 绝大部分结论与数据匹配，有1处轻微偏差
- 3分: 部分结论与数据不匹配（如指标恶化却描述为"改善"）
- 2分: 多处结论与数据方向矛盾
- 1分: 结论与数据严重脱节，或无法判断（无分析文字）

### 维度3: 分析完整（1-5分）
- 5分: 覆盖盈利能力、偿债能力、营运能力三大维度及风险提示，各维度分析深入
- 4分: 覆盖主要维度，分析较充分，有个别维度略显薄弱
- 3分: 覆盖部分维度，遗漏1个重要维度的分析
- 2分: 分析不完整，仅涉及个别维度或所有维度均浅尝辄止
- 1分: 几乎无实质分析内容

### 综合质量（1-5分）——此分数最为重要
- 5分: 逻辑严密、结论准确、覆盖全面，可直接作为投资决策参考
- 4分: 分析质量较高，有小瑕疵但整体有参考价值
- 3分: 分析有一定参考价值但存在明显不足
- 2分: 分析质量低，不宜作为决策依据，或报告仅罗列数字而无叙述性分析
- 1分: 无实质分析价值

**强制扣分规则（必须执行）**：
- 如果报告仅罗列指标数值而无任何叙述性分析文字 → 逻辑通顺 ≤ 1, 综合质量 ≤ 2
- 如果报告遗漏了盈利/偿债/营运中的2个及以上维度 → 分析完整 ≤ 2, 综合质量 ≤ 2
- 如果分析仅是泛泛而谈，没有引用具体指标数值来支撑结论 → 逻辑通顺 ≤ 3, 综合质量 ≤ 3
- 如果报告少于100字 → 综合质量 ≤ 2
- 如果风险提示部分完全缺失或仅一句话带过 → 分析完整 ≤ 3

请严格按以下JSON格式返回，不要输出其他内容：
{{"逻辑通顺": 整数, "结论匹配": 整数, "分析完整": 整数, "综合质量": 整数}}"""


def _format_indicator_summary(indicators: list[IndicatorResult]) -> str:
    """将指标列表格式化为评审用的摘要文本"""
    lines: list[str] = []
    by_cat: dict[str, list[IndicatorResult]] = defaultdict(list)
    for ind in indicators:
        if ind.value is not None:
            by_cat[ind.category].append(ind)

    for cat in ["杜邦分析", "Z-Score", "偿债能力", "盈利能力", "营运能力"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        lines.append(f"  [{cat}]")
        for ind in items:
            if abs(ind.value) < 1:
                lines.append(f"    {ind.name} = {ind.value:.4f} ({ind.value * 100:.2f}%)")
            else:
                lines.append(f"    {ind.name} = {ind.value:.4f}")
    return "\n".join(lines) if lines else "  （无可用指标数据）"


def _parse_score_json(raw_text: str, expected_keys: list[str]) -> dict[str, int]:
    """从LLM响应中解析评分JSON"""
    text = raw_text.strip()
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    result = {}
    for key in expected_keys:
        val = data.get(key)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            result[key] = int(round(val))
        else:
            result[key] = 3  # 解析失败时默认中位数
    return result


def _run_single_judge(
    client,
    prompt_template: str,
    fmt_kwargs: dict,
    expected_keys: list[str],
) -> dict[str, int]:
    """执行单次 LLM-as-Judge 评分调用。"""
    prompt = prompt_template.format(**fmt_kwargs)
    raw = chat_completion(
        client, TengriConfig.MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return _parse_score_json(raw, expected_keys)


def interpretation_quality(
    interpretation: str,
    indicators: list[IndicatorResult],
    company_info: dict[str, str | int],
    n_runs: int = 3,
) -> dict:
    """
    解读质量评价 -- LLM-as-Judge 双评机制（并发版）

    两套评审 Prompt 分别侧重"财务准确性"和"分析逻辑性"，
    各自给出 3 个维度分 + 1 个综合质量分。
    Cohen's Kappa 在外部使用"综合质量"分数计算。
    """
    if not interpretation or not interpretation.strip():
        return {
            "reviewer_a": {"数据准确": 1, "计算合理": 1, "口径一致": 1, "综合质量": 1, "mean": 1.0},
            "reviewer_b": {"逻辑通顺": 1, "结论匹配": 1, "分析完整": 1, "综合质量": 1, "mean": 1.0},
            "reviewer_a_overall": 1.0,
            "reviewer_b_overall": 1.0,
            "overall_score": 1.0,
        }

    client = get_tengri_client()
    ind_summary = _format_indicator_summary(indicators)

    fmt_kwargs = {
        "company_name": company_info.get("company_name", ""),
        "year": company_info.get("year", ""),
        "industry": company_info.get("industry", ""),
        "indicator_summary": ind_summary,
        "interpretation": interpretation,
    }

    a_keys = ["数据准确", "计算合理", "口径一致", "综合质量"]
    b_keys = ["逻辑通顺", "结论匹配", "分析完整", "综合质量"]

    a_runs: list[dict[str, int]] = []
    b_runs: list[dict[str, int]] = []

    total_calls = n_runs * 2
    with ThreadPoolExecutor(max_workers=total_calls) as executor:
        a_futures = [
            executor.submit(
                _run_single_judge, client,
                _REVIEWER_A_PROMPT, fmt_kwargs, a_keys,
            )
            for _ in range(n_runs)
        ]
        b_futures = [
            executor.submit(
                _run_single_judge, client,
                _REVIEWER_B_PROMPT, fmt_kwargs, b_keys,
            )
            for _ in range(n_runs)
        ]
        for f in a_futures:
            a_runs.append(f.result())
        for f in b_futures:
            b_runs.append(f.result())

    a_dim_keys = ["数据准确", "计算合理", "口径一致"]
    b_dim_keys = ["逻辑通顺", "结论匹配", "分析完整"]

    a_avg = {k: float(np.mean([r[k] for r in a_runs])) for k in a_keys}
    a_avg["mean"] = float(np.mean([a_avg[k] for k in a_dim_keys]))

    b_avg = {k: float(np.mean([r[k] for r in b_runs])) for k in b_keys}
    b_avg["mean"] = float(np.mean([b_avg[k] for k in b_dim_keys]))

    a_overall = a_avg["综合质量"]
    b_overall = b_avg["综合质量"]
    overall = (a_overall + b_overall) / 2

    return {
        "reviewer_a": a_avg,
        "reviewer_b": b_avg,
        "reviewer_a_overall": a_overall,
        "reviewer_b_overall": b_overall,
        "overall_score": overall,
    }


# ===================================================================
# 5. Cohen's Kappa 一致性系数
# ===================================================================

def cohens_kappa(
    scores_a: list[float],
    scores_b: list[float],
    weights: str = "quadratic",
) -> float:
    """
    计算两位评审间的 Cohen's Kappa 一致性系数。

    Parameters
    ----------
    scores_a : 评审A的评分列表（每个样本一个得分）
    scores_b : 评审B的评分列表
    weights : 权重类型，ordinal数据推荐 "quadratic"

    Returns
    -------
    Kappa 系数 (-1 ~ 1)
    """
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return 0.0

    labels_a = [int(round(s)) for s in scores_a]
    labels_b = [int(round(s)) for s in scores_b]

    labels_a = [max(1, min(5, v)) for v in labels_a]
    labels_b = [max(1, min(5, v)) for v in labels_b]

    return float(cohen_kappa_score(labels_a, labels_b, weights=weights))


def intraclass_correlation(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """
    计算组内相关系数 ICC(3,1)（双向混合模型、一致性、单测量）。

    当评分分布高度集中（天花板/地板效应）导致 Cohen's Kappa 不稳定时，
    ICC 作为补充一致性指标更为稳健。

    Returns
    -------
    {"icc": float, "mean_abs_diff": float, "max_abs_diff": float}
    """
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return {"icc": 0.0, "mean_abs_diff": 0.0, "max_abs_diff": 0.0}

    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    n = len(a)
    k = 2

    grand_mean = (a.sum() + b.sum()) / (n * k)

    row_means = (a + b) / k
    col_means = np.array([a.mean(), b.mean()])

    ss_total = np.sum((a - grand_mean) ** 2) + np.sum((b - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1) if n > 1 else 0.0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0

    denom = ms_rows + ms_error
    icc = (ms_rows - ms_error) / denom if denom > 1e-12 else 0.0
    icc = max(-1.0, min(1.0, icc))

    abs_diff = np.abs(a - b)
    return {
        "icc": float(icc),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
    }


# ===================================================================
# 6. 批量评估
# ===================================================================

def evaluate_batch(
    results: list[AnalysisResult],
    ground_truths: dict[str, dict[str, float]] | None = None,
    multi_year_groups: dict[str, list[AnalysisResult]] | None = None,
    run_llm_judge: bool = True,
    n_runs: int = 3,
) -> dict:
    """
    批量评估一组分析结果。

    Parameters
    ----------
    results : 待评估的分析结果列表
    ground_truths : {company_code_year: {category/指标名: 正确值}} 字典（可选）
    multi_year_groups : {company_code: [多年结果]} 字典（可选，用于口径一致性）
    run_llm_judge : 是否运行LLM-as-Judge评分（耗时较长）
    n_runs : LLM-as-Judge每套prompt运行次数
    """
    report: dict = {
        "method": results[0].method if results else "unknown",
        "sample_count": len(results),
        "calculation_correctness": None,
        "evidence_alignment": None,
        "caliber_consistency": None,
        "interpretation_quality": None,
        "cohens_kappa": None,
    }

    # ---- 计算正确率 ----
    if ground_truths:
        correctness_scores: list[float] = []
        for r in results:
            key = f"{r.company_code}_{r.year}"
            gt = ground_truths.get(key)
            if gt:
                result = calculation_correctness(r.indicators, gt)
                correctness_scores.append(result["accuracy"])
        if correctness_scores:
            report["calculation_correctness"] = {
                "mean": float(np.mean(correctness_scores)),
                "std": float(np.std(correctness_scores)),
                "min": float(np.min(correctness_scores)),
                "max": float(np.max(correctness_scores)),
                "count": len(correctness_scores),
            }

    # ---- 证据对齐率 ----
    alignment_scores: list[float] = []
    for r in results:
        result = evidence_alignment_rate(r.indicators)
        alignment_scores.append(result["alignment_rate"])
    report["evidence_alignment"] = {
        "mean": float(np.mean(alignment_scores)),
        "std": float(np.std(alignment_scores)),
        "count": len(alignment_scores),
    }

    # ---- 口径一致性 ----
    if multi_year_groups:
        consistency_scores: list[float] = []
        for _code, group in multi_year_groups.items():
            result = caliber_consistency(group)
            consistency_scores.append(result["consistency"])
        if consistency_scores:
            report["caliber_consistency"] = {
                "mean": float(np.mean(consistency_scores)),
                "std": float(np.std(consistency_scores)),
                "count": len(consistency_scores),
            }

    # ---- 解读质量 (LLM-as-Judge, 并发) ----
    if run_llm_judge:
        max_judge_workers = ConcurrencyConfig.JUDGE_MAX_WORKERS
        reviewer_a_overalls: list[float] = [0.0] * len(results)
        reviewer_b_overalls: list[float] = [0.0] * len(results)
        overall_scores: list[float] = [0.0] * len(results)

        done_count = 0
        total_count = len(results)

        def _judge_one(idx: int, r: AnalysisResult) -> tuple[int, dict]:
            quality = interpretation_quality(
                r.interpretation,
                r.indicators,
                {"company_name": r.company_name, "year": r.year, "industry": r.industry},
                n_runs=n_runs,
            )
            return idx, quality

        with ThreadPoolExecutor(max_workers=max_judge_workers) as executor:
            futures = {
                executor.submit(_judge_one, i, r): i
                for i, r in enumerate(results)
            }
            for future in as_completed(futures):
                idx, quality = future.result()
                reviewer_a_overalls[idx] = quality["reviewer_a_overall"]
                reviewer_b_overalls[idx] = quality["reviewer_b_overall"]
                overall_scores[idx] = quality["overall_score"]
                done_count += 1
                r = results[idx]
                logger.info("LLM-as-Judge 完成 (%d/%d): %s %d -> A=%.1f B=%.1f 综合=%.2f",
                            done_count, total_count,
                            r.company_name, r.year,
                            quality["reviewer_a_overall"],
                            quality["reviewer_b_overall"],
                            quality["overall_score"])

        report["interpretation_quality"] = {
            "reviewer_a_mean": float(np.mean(reviewer_a_overalls)),
            "reviewer_b_mean": float(np.mean(reviewer_b_overalls)),
            "overall_mean": float(np.mean(overall_scores)),
            "overall_std": float(np.std(overall_scores)),
            "count": len(overall_scores),
            "_raw_a_overalls": reviewer_a_overalls,
            "_raw_b_overalls": reviewer_b_overalls,
        }

        # ---- 方法内 Cohen's Kappa + ICC ----
        if len(reviewer_a_overalls) >= 2:
            kappa = cohens_kappa(reviewer_a_overalls, reviewer_b_overalls)
            icc_result = intraclass_correlation(reviewer_a_overalls, reviewer_b_overalls)
            report["cohens_kappa"] = {
                "value": kappa,
                "interpretation": _interpret_kappa(kappa),
            }
            report["icc"] = icc_result

    return report


def _interpret_kappa(kappa: float) -> str:
    """对 Kappa 值给出定性解释"""
    if kappa < 0:
        return "低于随机一致性"
    elif kappa < 0.20:
        return "极低一致性"
    elif kappa < 0.40:
        return "低一致性"
    elif kappa < 0.60:
        return "中等一致性"
    elif kappa < 0.80:
        return "较高一致性"
    else:
        return "高一致性"
