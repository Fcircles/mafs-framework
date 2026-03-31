"""基线2: 单一LLM方法 -- 直接输入年报文本生成分析结果（不采用多Agent协作）"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from baselines.base import BaseBaseline
from config import TengriConfig
from models import AnalysisResult, IndicatorResult
from utils.llm_client import get_tengri_client, chat_completion
from utils.pdf_parser import parse_annual_report, get_pages_text_with_metadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 指标类别列表（用于 Prompt 和结果解析）
# ---------------------------------------------------------------------------

REQUIRED_INDICATORS = {
    "杜邦分析": ["ROE", "销售净利率", "资产周转率", "权益乘数"],
    "Z-Score": ["Z-Score"],
    "偿债能力": ["流动比率", "速动比率", "资产负债率"],
    "盈利能力": ["毛利率", "净利率", "ROA", "ROE"],
    "营运能力": ["应收账款周转率", "存货周转率", "总资产周转率"],
}

# ---------------------------------------------------------------------------
# 系统 & 用户 Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一位专业的财务分析师。你将收到一份上市公司年报中与财务报表相关的文本内容。
请根据文本中的财务数据，完成以下任务：
1. 提取关键财务科目数值
2. 计算指定的财务指标
3. 给出简要的财务分析解读

请严格按照指定的JSON格式返回结果，不要添加任何多余的文字。"""

_USER_PROMPT_TEMPLATE = """\
以下是{company_name}（股票代码: {company_code}）{year}年年报中的财务报表相关内容：

{financial_text}

请完成以下分析任务：

1. 杜邦分析体系：计算ROE并分解为销售净利率、资产周转率、权益乘数
2. Altman Z-Score：根据行业（{industry}）选用合适版本{zscore_note}
3. 偿债能力指标：流动比率、速动比率、资产负债率
4. 盈利能力指标：毛利率、净利率、ROA、ROE
5. 营运能力指标：应收账款周转率、存货周转率、总资产周转率
6. 综合解读：基于上述指标给出200-400字的财务分析解读

请严格按以下JSON格式返回（不要输出其他内容）：
{{
  "indicators": [
    {{
      "name": "指标名称",
      "value": 数值或null,
      "formula": "使用的计算公式",
      "category": "杜邦分析/Z-Score/偿债能力/盈利能力/营运能力",
      "source_text": "计算所依据的原始数据描述"
    }}
  ],
  "interpretation": "综合财务分析解读文本"
}}"""


def _build_zscore_note(industry: str) -> str:
    if industry == "制造业":
        return "（制造业用原始Z-Score公式: Z=1.2X1+1.4X2+3.3X3+0.6X4+1.0X5）"
    return "（服务类企业用Z''修正模型: Z''=6.56X1+3.26X2+6.72X3+1.05X4）"


def _extract_financial_pages_text(
    pdf_path: str | Path,
    max_chars: int = 30000,
) -> tuple[str, list[int]]:
    """提取年报中与财务报表相关的页面文本"""
    report = parse_annual_report(pdf_path)
    pages_data = get_pages_text_with_metadata(report)

    financial_pages = [
        p for p in pages_data
        if p["has_tables"] and p["table_types"]
    ]

    if not financial_pages:
        financial_pages = [p for p in pages_data if p["has_tables"]]

    if not financial_pages:
        financial_pages = pages_data

    text_parts: list[str] = []
    page_nums: list[int] = []
    total_len = 0
    for p in financial_pages:
        t = p["text"].strip()
        if not t:
            continue
        if total_len + len(t) > max_chars:
            break
        text_parts.append(f"--- 第{p['page_number']}页 ---\n{t}")
        page_nums.append(p["page_number"])
        total_len += len(t)

    return "\n\n".join(text_parts), page_nums


def _parse_llm_json(raw_text: str) -> dict:
    """从LLM响应中提取JSON（处理markdown代码块等格式）"""
    text = raw_text.strip()

    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        text = md_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("无法解析LLM JSON响应，返回空结果")
    return {"indicators": [], "interpretation": raw_text}


# ---------------------------------------------------------------------------
# 核心分析函数
# ---------------------------------------------------------------------------

def analyze_with_llm(
    client: OpenAI,
    model: str,
    pdf_path: str | Path,
    company_code: str,
    company_name: str,
    year: int,
    industry: str,
    *,
    method_name: str = "single_llm",
    max_chars: int = 30000,
    temperature: float = 0.1,
) -> AnalysisResult:
    """
    用单一LLM直接分析年报。

    这是一个参数化的公共函数，接收 client 和 model 参数。
    """
    financial_text, page_nums = _extract_financial_pages_text(pdf_path, max_chars)

    if not financial_text.strip():
        return AnalysisResult(
            company_code=company_code,
            company_name=company_name,
            year=year,
            industry=industry,
            method=method_name,
            indicators=[],
            interpretation="未能从年报PDF中提取到有效的财务文本。",
        )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        company_code=company_code,
        year=year,
        industry=industry,
        zscore_note=_build_zscore_note(industry),
        financial_text=financial_text,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = chat_completion(
        client, model, messages,
        temperature=temperature,
    )

    parsed = _parse_llm_json(raw_response)

    indicators: list[IndicatorResult] = []
    for item in parsed.get("indicators", []):
        val = item.get("value")
        if val is not None:
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = None
        indicators.append(IndicatorResult(
            name=item.get("name", ""),
            value=val,
            formula=item.get("formula", ""),
            source_page=page_nums[0] if page_nums else None,
            source_text=item.get("source_text") or "",
            category=item.get("category", ""),
        ))

    interpretation = parsed.get("interpretation", "")

    return AnalysisResult(
        company_code=company_code,
        company_name=company_name,
        year=year,
        industry=industry,
        method=method_name,
        indicators=indicators,
        interpretation=interpretation,
        metadata={
            "model": model,
            "pages_used": page_nums,
            "text_length": len(financial_text),
        },
    )


# ---------------------------------------------------------------------------
# 基线类
# ---------------------------------------------------------------------------

class SingleLLMBaseline(BaseBaseline):
    """基线2: Tengri单一LLM直接分析"""

    def __init__(self):
        self._client: OpenAI | None = None

    @property
    def name(self) -> str:
        return "single_llm"

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = get_tengri_client()
        return self._client

    def analyze(
        self,
        pdf_path: str | Path,
        company_code: str,
        company_name: str,
        year: int,
        industry: str,
    ) -> AnalysisResult:
        return analyze_with_llm(
            client=self.client,
            model=TengriConfig.MODEL,
            pdf_path=pdf_path,
            company_code=company_code,
            company_name=company_name,
            year=year,
            industry=industry,
            method_name=self.name,
        )
