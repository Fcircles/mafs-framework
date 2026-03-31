"""解读生成Agent -- 基于计算结果生成结构化财务分析报告"""

from __future__ import annotations

import json
import logging
import time

from agents import PipelineState
from utils.llm_client import get_client_for_provider, get_model_for_provider, chat_completion, EmptyResponseError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5

SYSTEM_PROMPT = """\
你是一名专业财务分析师，擅长基于年报数据撰写结构化的财务分析报告。
请严格遵守以下规则：
1. 只能引用"财务指标计算结果"中出现的比率数值和"年报原文证据"中出现的文字，严禁编造、推算或引用未提供的绝对金额数字。
2. 每项结论后用方括号标注数据来源，格式如 [第X页] 或 [数据来源: 资产负债表]。
3. 当只有单一年度数据、没有上年同期对比时，只能描述指标的绝对水平（如"处于较高/较低水平"），严禁使用"改善""恶化""提升""下降"等趋势性判断词，除非证据中明确提供了同比数据。
4. Z-Score/Z''-Score 区间判断须与指标中的 zone 判定结论严格一致，不得自行重新判定区间。
   制造业 Z-Score 参考阈值：>2.99 安全区，1.81~2.99 灰色区，<1.81 危险区。
   非制造业 Z''-Score 参考阈值：>2.60 安全区，1.10~2.60 灰色区，<1.10 危险区。
   Z-Score/Z''-Score为负数时必须明确指出处于危险区。
5. 使用专业术语，语言简洁准确，适合投资者和审计人员阅读。
"""


def _format_indicators(indicators: dict) -> str:
    """将指标字典格式化为结构化文本。"""
    sections: list[str] = []

    for category, data in indicators.items():
        if category == "计算缺失项":
            if data:
                sections.append(f"【未计算指标】\n" + "\n".join(f"  - {m}" for m in data))
            continue
        if not isinstance(data, dict):
            continue

        lines = [f"【{category}】"]
        for k, v in data.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _format_evidence(evidence_map: list[dict], max_per_indicator: int = 2) -> str:
    """将证据列表格式化为参考文本，每个指标最多保留 max_per_indicator 条。"""
    if not evidence_map:
        return "（无可用证据）"

    seen: dict[str, int] = {}
    lines: list[str] = []

    for ev in evidence_map:
        key = f"{ev['category']}/{ev['indicator_name']}"
        count = seen.get(key, 0)
        if count >= max_per_indicator:
            continue
        seen[key] = count + 1

        page = ev.get("page_number", "未知")
        section = ev.get("section_title", "")
        text = ev.get("source_text", "")[:200]
        lines.append(f"[{key} | 第{page}页 | {section}]\n  {text}")

    return "\n\n".join(lines) if lines else "（无可用证据）"


def _format_extracted_values(extracted: dict) -> str:
    """格式化已验证的原始财务数据。"""
    if not extracted:
        return "（无可用数据）"
    lines: list[str] = []
    for k, v in extracted.items():
        if v is not None:
            if abs(v) >= 1e8:
                lines.append(f"  {k}: {v/1e8:,.2f}亿元")
            else:
                lines.append(f"  {k}: {v:,.0f}元")
    return "\n".join(lines) if lines else "（无可用数据）"


def _build_user_prompt(state: PipelineState) -> str:
    """构建用户 prompt。"""
    company = state.get("company_name", "未知公司")
    code = state.get("company_code", "")
    industry = state.get("industry", "")
    year = state.get("year", "")
    indicators = state.get("indicators", {})
    evidence_map = state.get("evidence_map", [])
    extracted = state.get("extracted_values", {})

    indicator_text = _format_indicators(indicators)
    evidence_text = _format_evidence(evidence_map)
    extracted_text = _format_extracted_values(extracted)

    return f"""\
请基于以下财务指标数据和年报原文证据，为该公司撰写一份结构化财务分析报告。

## 公司信息
- 公司名称: {company}（{code}）
- 所属行业: {industry}
- 报告期间: {year}年度

## 财务指标计算结果（权威数据，必须以此为准）
{indicator_text}

## 已验证的合并报表原始数据（如需引用绝对金额，只能使用以下数值）
{extracted_text}

## 年报原文证据（仅供理解上下文，其中的数值可能来自母公司报表或明细科目，不得直接引用其中的绝对金额）
{evidence_text}

## 输出要求
请按以下结构撰写报告：

一、企业概况与分析期间
（简要介绍公司及报告期间）

二、盈利能力分析
（基于杜邦分解结果，分析 ROE 的驱动因素：销售净利率、资产周转率、权益乘数。引用具体数值和证据。）

三、偿债能力分析
（分析流动比率、速动比率、资产负债率。对 Z-Score/Z''-Score 结果，必须直接使用上方指标数据中的 zone 判定结论。如指标中有 model_caveat 字段，须在分析中引用该局限性说明。）

四、营运能力分析
（分析应收账款周转率、存货周转率、总资产周转率。评估资产运营效率。）

五、综合评价与风险提示
（综合各维度分析结果，给出整体评价。指出主要风险点和关注事项。）
"""


def interpretation_generator_node(state: PipelineState) -> dict:
    """解读生成节点：调用 LLM 生成结构化财务分析报告。"""

    indicators = state.get("indicators", {})
    errors: list[str] = list(state.get("errors", []))

    if not indicators or all(
        not isinstance(v, dict) for v in indicators.values()
    ):
        msg = "无有效指标数据，跳过解读生成"
        logger.warning(msg)
        errors.append(msg)
        return {"interpretation": "", "errors": errors}

    user_prompt = _build_user_prompt(state)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    _provider = state.get("llm_provider", "tengri")
    client = get_client_for_provider(_provider)
    _model = state.get("llm_model") or get_model_for_provider(_provider)
    interpretation = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            interpretation = chat_completion(
                client,
                _model,
                messages,
                temperature=0.3,
            )
            logger.info("解读生成完成 (第%d次尝试), 长度: %d 字符",
                         attempt, len(interpretation))
            break
        except EmptyResponseError as exc:
            msg = f"LLM 返回空内容 (第{attempt}次): {exc}"
            logger.warning(msg)
            if attempt == MAX_RETRIES:
                errors.append(msg)
            else:
                time.sleep(RETRY_DELAY)
        except Exception as exc:
            msg = f"LLM 调用失败 (第{attempt}次): {exc}"
            logger.warning(msg)
            if attempt == MAX_RETRIES:
                errors.append(msg)
            else:
                time.sleep(RETRY_DELAY)

    return {"interpretation": interpretation, "errors": errors}
