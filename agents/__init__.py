"""多智能体模块 -- 基于LangGraph的年报分析框架"""

from __future__ import annotations

import re
import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """LangGraph 状态图的统一数据契约。

    所有 Agent 节点函数的签名为 ``(state: PipelineState) -> dict``，
    返回值仅包含本节点需要更新的键。
    """

    # -- 输入 --
    pdf_path: str
    company_code: str
    company_name: str
    industry: str
    year: int

    # -- 文档解析 Agent 输出 --
    parsed_report: Any
    financial_tables: dict
    extracted_values: dict
    vector_store: Any

    # -- 指标计算 Agent 输出 --
    indicators: dict

    # -- 证据对齐 Agent 输出 --
    evidence_map: list

    # -- 解读生成 Agent 输出 --
    interpretation: str

    # -- 一致性自检 Agent 输出 --
    consistency_report: dict
    consistency_passed: bool

    # -- 流程控制 --
    retry_count: int
    errors: list

    # -- LLM 配置 --
    llm_provider: str   # 默认 "tengri"
    llm_model: str      # 模型名称覆盖，空字符串表示使用默认


# ---------------------------------------------------------------------------
# 公共常量 -- 财务报表项目别名映射
# ---------------------------------------------------------------------------

ITEM_ALIASES: dict[str, list[str]] = {
    # 利润表 -- 精确匹配，不含"营业总收入/总成本"以避免与含金融
    # 子公司的合并利润表中的"一、营业总收入""二、营业总成本"混淆。
    # 前缀"一、""减："等会被 _PREFIX_RE 自动剥离，不需要列为别名。
    "营业收入": ["营业收入"],
    "营业成本": ["营业成本"],
    "净利润": ["净利润"],
    "利润总额": ["利润总额"],
    "财务费用": ["财务费用"],
    "所得税费用": ["所得税费用"],

    # 资产负债表
    "资产总计": ["资产总计", "资产合计", "资产总额"],
    "流动资产合计": ["流动资产合计"],
    "流动负债合计": ["流动负债合计"],
    "负债合计": ["负债合计", "负债总计", "负债总额"],
    "所有者权益合计": [
        "所有者权益合计", "股东权益合计",
        "所有者权益（或股东权益）合计",
        "所有者权益(或股东权益)合计",
        "负债和所有者权益总计",
        "负债及所有者权益总计",
    ],
    "存货": ["存货"],
    "应收账款": ["应收账款"],
    "未分配利润": ["未分配利润"],
    "盈余公积": ["盈余公积"],

    # 期初值（用于计算平均值）
    "期初资产总计": ["资产总计", "资产合计", "资产总额"],
    "期初所有者权益合计": [
        "所有者权益合计", "股东权益合计",
        "所有者权益（或股东权益）合计",
        "所有者权益(或股东权益)合计",
    ],
    "期初应收账款": ["应收账款"],
    "期初存货": ["存货"],
}

# 当精确匹配找不到时的备选别名（仅在 fallback 阶段使用）
ITEM_FALLBACK_ALIASES: dict[str, list[str]] = {
    "营业收入": ["营业总收入"],
    "营业成本": ["营业总成本"],
    "净利润": ["归属于母公司所有者的净利润", "归属于母公司股东的净利润"],
    "应收账款": ["应收票据及应收账款"],
    "期初应收账款": ["应收票据及应收账款"],
}


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[^\d.\-]")


def parse_number(raw: str | None) -> float | None:
    """将年报中的数值字符串解析为 float。

    处理逗号分隔、括号负数 ``(123)`` -> ``-123``、空白等情况。
    返回 ``None`` 表示解析失败。
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("-", "--", "—", ""):
        return None

    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    elif s.startswith("-"):
        negative = True
        s = s[1:]

    s = _NUM_RE.sub("", s)
    if not s:
        return None

    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def safe_get(values: dict, key: str, default: float = 0.0) -> float:
    """从 extracted_values 中安全取值，缺失时返回 default。"""
    v = values.get(key)
    if v is None:
        return default
    return float(v)
