"""基线1: 传统规则/词典方法 -- 基于关键词和正则表达式提取财务数据"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import pandas as pd

from baselines.base import BaseBaseline
from models import AnalysisResult, IndicatorResult
from utils.pdf_parser import (
    parse_annual_report,
    extract_financial_data,
    ParsedReport,
)
from utils.financial_formulas import (
    dupont_analysis,
    altman_zscore_manufacturing,
    altman_zscore_service,
    solvency_ratios,
    profitability_ratios,
    efficiency_ratios,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 财务科目正则匹配模式
# ---------------------------------------------------------------------------

_BALANCE_SHEET_PATTERNS: dict[str, re.Pattern] = {
    "流动资产合计": re.compile(r"流动资产[合小]计"),
    "流动负债合计": re.compile(r"流动负债[合小]计"),
    "总资产": re.compile(r"资产[总合]计"),
    "总负债": re.compile(r"负债[总合]计"),
    "存货": re.compile(r"^[\s]*存\s*货[\s]*$"),
    "应收账款": re.compile(r"应收[账帐]款"),
    "所有者权益合计": re.compile(r"(?:所有者权益|股东权益).*[合小]计"),
    "盈余公积": re.compile(r"盈余公积"),
    "未分配利润": re.compile(r"未分配利润"),
}

_INCOME_PATTERNS: dict[str, re.Pattern] = {
    "营业收入": re.compile(r"营业(?:总)?收入"),
    "营业成本": re.compile(r"营业(?:总)?成本"),
    "利润总额": re.compile(r"利润总额"),
    "净利润": re.compile(r"净利润"),
    "财务费用": re.compile(r"^[\s]*财务费用"),
}


def _parse_number(text: str | None) -> float | None:
    """将财务报表单元格文本解析为浮点数"""
    if not text:
        return None
    text = str(text).strip()
    if text in ("", "-", "--", "—", "——", "/", "不适用"):
        return None

    negative = text.startswith("(") or text.startswith("\uff08")
    text = (
        text.replace("(", "").replace(")", "")
        .replace("\uff08", "").replace("\uff09", "")
        .replace(",", "").replace("\uff0c", "")
        .replace(" ", "")
    )
    try:
        value = float(text)
        return -value if negative else value
    except ValueError:
        return None


def _search_df(
    df: pd.DataFrame,
    pattern: re.Pattern,
    value_col: int = 1,
) -> float | None:
    """在 DataFrame 第一列中搜索匹配项，返回对应值列的数值"""
    if df.empty or df.shape[1] <= value_col:
        return None
    for row_idx in range(len(df)):
        cell = str(df.iloc[row_idx, 0]).strip()
        if pattern.search(cell):
            val = _parse_number(df.iloc[row_idx, value_col])
            if val is not None:
                return val
    return None


def _search_df_with_page(
    df: pd.DataFrame,
    pattern: re.Pattern,
    value_col: int = 1,
) -> tuple[float | None, str]:
    """搜索并同时返回匹配的原文行文本（用于证据溯源）"""
    if df.empty or df.shape[1] <= value_col:
        return None, ""
    for row_idx in range(len(df)):
        cell = str(df.iloc[row_idx, 0]).strip()
        if pattern.search(cell):
            val = _parse_number(df.iloc[row_idx, value_col])
            if val is not None:
                row_text = " | ".join(str(c) for c in df.iloc[row_idx])
                return val, row_text
    return None, ""


def _find_statement_pages(report: ParsedReport, stmt_type: str) -> list[int]:
    """返回指定财务报表所在的页码列表"""
    pages = []
    for t in report.tables:
        if t.table_type == stmt_type:
            pages.append(t.page_number)
    return sorted(set(pages))


class RuleBasedBaseline(BaseBaseline):
    """基线1: 纯规则+公式计算方法，无LLM调用"""

    @property
    def name(self) -> str:
        return "rule_based"

    def analyze(
        self,
        pdf_path: str | Path,
        company_code: str,
        company_name: str,
        year: int,
        industry: str,
    ) -> AnalysisResult:
        pdf_path = Path(pdf_path)
        report = parse_annual_report(pdf_path)
        financial_dfs = extract_financial_data(report)

        raw = self._extract_items(financial_dfs, report)
        indicators = self._calculate_all(raw, industry, report)
        interpretation = self._build_interpretation(indicators, company_name, year)

        return AnalysisResult(
            company_code=company_code,
            company_name=company_name,
            year=year,
            industry=industry,
            method=self.name,
            indicators=indicators,
            interpretation=interpretation,
            metadata={
                "extracted_raw": {k: v for k, v in raw.items() if v is not None},
            },
        )

    # ------------------------------------------------------------------
    # 数据提取
    # ------------------------------------------------------------------

    def _extract_items(
        self,
        financial_dfs: dict[str, pd.DataFrame],
        report: ParsedReport,
    ) -> dict[str, float | None]:
        """从三大报表 DataFrame 中提取所有所需科目"""
        raw: dict[str, float | None] = {}

        bs_df = financial_dfs.get("资产负债表", pd.DataFrame())
        for key, pat in _BALANCE_SHEET_PATTERNS.items():
            raw[key] = _search_df(bs_df, pat)

        is_df = financial_dfs.get("利润表", pd.DataFrame())
        for key, pat in _INCOME_PATTERNS.items():
            raw[key] = _search_df(is_df, pat)

        # 毛利润 = 营业收入 - 营业成本
        rev = raw.get("营业收入")
        cost = raw.get("营业成本")
        raw["毛利润"] = (rev - cost) if (rev is not None and cost is not None) else None

        # 留存收益 = 盈余公积 + 未分配利润
        sp = raw.get("盈余公积")
        up = raw.get("未分配利润")
        raw["留存收益"] = ((sp or 0) + (up or 0)) if (sp is not None or up is not None) else None

        # 营运资本 = 流动资产 - 流动负债
        ca = raw.get("流动资产合计")
        cl = raw.get("流动负债合计")
        raw["营运资本"] = (ca - cl) if (ca is not None and cl is not None) else None

        logger.info("规则提取结果: %d/%d 项成功",
                     sum(1 for v in raw.values() if v is not None), len(raw))
        return raw

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def _calculate_all(
        self,
        raw: dict[str, float | None],
        industry: str,
        report: ParsedReport,
    ) -> list[IndicatorResult]:
        indicators: list[IndicatorResult] = []

        bs_pages = _find_statement_pages(report, "资产负债表")
        is_pages = _find_statement_pages(report, "利润表")
        bs_page = bs_pages[0] if bs_pages else None
        is_page = is_pages[0] if is_pages else None

        indicators.extend(self._calc_dupont(raw, bs_page, is_page))
        indicators.extend(self._calc_zscore(raw, industry, bs_page, is_page))
        indicators.extend(self._calc_solvency(raw, bs_page))
        indicators.extend(self._calc_profitability(raw, bs_page, is_page))
        indicators.extend(self._calc_efficiency(raw, bs_page, is_page))

        return indicators

    def _calc_dupont(self, raw, bs_page, is_page) -> list[IndicatorResult]:
        ni = raw.get("净利润")
        rev = raw.get("营业收入")
        ta = raw.get("总资产")
        eq = raw.get("所有者权益合计")
        if None in (ni, rev, ta, eq):
            return []

        result = dupont_analysis(ni, rev, ta, eq)
        page = is_page or bs_page
        source = f"净利润={ni}, 营业收入={rev}, 总资产={ta}, 所有者权益={eq}"
        return [
            IndicatorResult(
                name="ROE", value=result["ROE"],
                formula="净利润/营业收入 * 营业收入/总资产 * 总资产/所有者权益",
                source_page=page, source_text=source, category="杜邦分析",
            ),
            IndicatorResult(
                name="销售净利率", value=result["销售净利率"],
                formula="净利润/营业收入",
                source_page=is_page, source_text=source, category="杜邦分析",
            ),
            IndicatorResult(
                name="资产周转率", value=result["资产周转率"],
                formula="营业收入/总资产",
                source_page=page, source_text=source, category="杜邦分析",
            ),
            IndicatorResult(
                name="权益乘数", value=result["权益乘数"],
                formula="总资产/所有者权益",
                source_page=bs_page, source_text=source, category="杜邦分析",
            ),
        ]

    def _calc_zscore(self, raw, industry, bs_page, is_page) -> list[IndicatorResult]:
        wc = raw.get("营运资本")
        re_ = raw.get("留存收益")
        tp = raw.get("利润总额")
        fe = raw.get("财务费用", 0) or 0
        ebit = (tp + fe) if tp is not None else None
        ta = raw.get("总资产")
        tl = raw.get("总负债")
        eq = raw.get("所有者权益合计")
        rev = raw.get("营业收入")

        if None in (wc, re_, ebit, ta, tl):
            return []

        page = bs_page or is_page
        source = (f"营运资本={wc}, 留存收益={re_}, EBIT={ebit}, "
                  f"总资产={ta}, 总负债={tl}")

        if industry == "制造业":
            if rev is None:
                return []
            result = altman_zscore_manufacturing(
                wc, re_, ebit, eq or 0, tl, ta, rev,
            )
            score_key = "Z-Score"
        else:
            result = altman_zscore_service(wc, re_, ebit, eq or 0, tl, ta)
            score_key = "Z''-Score"

        indicators = [
            IndicatorResult(
                name=score_key, value=result[score_key],
                formula="Altman Z-Score" if industry == "制造业" else "Altman Z''-Score",
                source_page=page, source_text=source,
                category="Z-Score",
            ),
            IndicatorResult(
                name="Z-Score区域", value=None,
                formula=result["zone"],
                source_page=page, source_text=source,
                category="Z-Score",
            ),
        ]
        return indicators

    def _calc_solvency(self, raw, bs_page) -> list[IndicatorResult]:
        ca = raw.get("流动资产合计")
        cl = raw.get("流动负债合计")
        inv = raw.get("存货", 0) or 0
        ta = raw.get("总资产")
        tl = raw.get("总负债")

        if None in (ca, cl, ta, tl):
            return []

        result = solvency_ratios(ca, cl, inv, ta, tl)
        source = f"流动资产={ca}, 流动负债={cl}, 存货={inv}, 总资产={ta}, 总负债={tl}"
        return [
            IndicatorResult(
                name=k, value=v,
                formula={"流动比率": "流动资产/流动负债",
                         "速动比率": "(流动资产-存货)/流动负债",
                         "资产负债率": "总负债/总资产"}.get(k, ""),
                source_page=bs_page, source_text=source,
                category="偿债能力",
            )
            for k, v in result.items()
        ]

    def _calc_profitability(self, raw, bs_page, is_page) -> list[IndicatorResult]:
        ni = raw.get("净利润")
        rev = raw.get("营业收入")
        gp = raw.get("毛利润")
        ta = raw.get("总资产")
        eq = raw.get("所有者权益合计")

        if None in (ni, rev, ta, eq):
            return []
        if gp is None:
            gp = 0.0

        result = profitability_ratios(ni, rev, gp, ta, eq)
        source = f"净利润={ni}, 营业收入={rev}, 毛利润={gp}, 总资产={ta}, 权益={eq}"
        formulas = {
            "毛利率": "(营业收入-营业成本)/营业收入",
            "净利率": "净利润/营业收入",
            "ROA": "净利润/总资产",
            "ROE": "净利润/所有者权益",
        }
        return [
            IndicatorResult(
                name=k, value=v,
                formula=formulas.get(k, ""),
                source_page=is_page or bs_page, source_text=source,
                category="盈利能力",
            )
            for k, v in result.items()
        ]

    def _calc_efficiency(self, raw, bs_page, is_page) -> list[IndicatorResult]:
        rev = raw.get("营业收入")
        recv = raw.get("应收账款")
        cogs = raw.get("营业成本")
        inv = raw.get("存货")
        ta = raw.get("总资产")

        if rev is None or ta is None:
            return []

        result = efficiency_ratios(
            rev,
            recv if recv else 1.0,
            cogs if cogs else 0.0,
            inv if inv else 1.0,
            ta,
        )
        source = f"营业收入={rev}, 应收账款={recv}, 营业成本={cogs}, 存货={inv}, 总资产={ta}"
        formulas = {
            "应收账款周转率": "营业收入/应收账款",
            "存货周转率": "营业成本/存货",
            "总资产周转率": "营业收入/总资产",
        }
        has_data = {
            "应收账款周转率": recv is not None,
            "存货周转率": cogs is not None and inv is not None,
            "总资产周转率": True,
        }
        return [
            IndicatorResult(
                name=k, value=v if has_data[k] else None,
                formula=formulas.get(k, ""),
                source_page=is_page or bs_page, source_text=source,
                category="营运能力",
            )
            for k, v in result.items()
            if has_data.get(k, True)
        ]

    # ------------------------------------------------------------------
    # 结构化解读文本
    # ------------------------------------------------------------------

    @staticmethod
    def _build_interpretation(
        indicators: list[IndicatorResult],
        company_name: str,
        year: int,
    ) -> str:
        parts: list[str] = [f"{company_name} {year}年年报财务分析（规则提取）\n"]

        categories = ["杜邦分析", "Z-Score", "偿债能力", "盈利能力", "营运能力"]
        for cat in categories:
            items = [i for i in indicators if i.category == cat]
            if not items:
                continue
            parts.append(f"【{cat}】")
            for item in items:
                if item.value is not None:
                    if abs(item.value) < 1:
                        parts.append(f"  {item.name} = {item.value:.4f} ({item.value * 100:.2f}%)")
                    else:
                        parts.append(f"  {item.name} = {item.value:.4f}")
                elif item.formula:
                    parts.append(f"  {item.name}: {item.formula}")
            parts.append("")

        if not any(i.value is not None for i in indicators):
            parts.append("未能从年报中提取到足够的财务数据进行计算。")

        return "\n".join(parts)
