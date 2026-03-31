"""指标计算Agent -- 基于财务模型进行标准化计算"""

from __future__ import annotations

import logging

from agents import PipelineState, safe_get
from utils.financial_formulas import (
    dupont_analysis,
    altman_zscore_manufacturing,
    altman_zscore_service,
    solvency_ratios,
    profitability_ratios,
    efficiency_ratios,
)

logger = logging.getLogger(__name__)


def _compute_derived_values(v: dict) -> dict:
    """从原始财务数值计算衍生变量。"""
    current_assets = safe_get(v, "流动资产合计")
    current_liabilities = safe_get(v, "流动负债合计")
    total_assets = safe_get(v, "资产总计")
    prior_total_assets = safe_get(v, "期初资产总计")
    equity = safe_get(v, "所有者权益合计")
    prior_equity = safe_get(v, "期初所有者权益合计")
    revenue = safe_get(v, "营业收入")
    cost = safe_get(v, "营业成本")
    profit_before_tax = safe_get(v, "利润总额")
    finance_expense = safe_get(v, "财务费用")
    undistributed = safe_get(v, "未分配利润")
    surplus_reserve = safe_get(v, "盈余公积")
    receivables = safe_get(v, "应收账款")
    prior_receivables = safe_get(v, "期初应收账款")
    inventory = safe_get(v, "存货")
    prior_inventory = safe_get(v, "期初存货")

    avg_total_assets = (total_assets + prior_total_assets) / 2 if prior_total_assets else total_assets
    avg_equity = (equity + prior_equity) / 2 if prior_equity else equity
    avg_receivables = (receivables + prior_receivables) / 2 if prior_receivables else receivables
    avg_inventory = (inventory + prior_inventory) / 2 if prior_inventory else inventory

    return {
        "working_capital": current_assets - current_liabilities,
        "retained_earnings": undistributed + surplus_reserve,
        "ebit": profit_before_tax + finance_expense,
        "gross_profit": revenue - cost,
        "avg_total_assets": avg_total_assets,
        "avg_equity": avg_equity,
        "avg_receivables": avg_receivables,
        "avg_inventory": avg_inventory,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "total_assets": total_assets,
        "total_liabilities": safe_get(v, "负债合计"),
        "equity": equity,
        "inventory": inventory,
        "revenue": revenue,
        "cost": cost,
        "net_income": safe_get(v, "净利润"),
    }


def indicator_calculator_node(state: PipelineState) -> dict:
    """指标计算节点：基于提取的财务数值计算全部指标。"""

    extracted = state.get("extracted_values", {})
    errors: list[str] = list(state.get("errors", []))

    if not extracted or all(v is None for v in extracted.values()):
        msg = "无可用财务数值，跳过指标计算"
        logger.warning(msg)
        errors.append(msg)
        return {"indicators": {}, "errors": errors}

    d = _compute_derived_values(extracted)
    missing: list[str] = []
    indicators: dict = {}

    # 1) 杜邦分析
    try:
        if d["net_income"] and d["revenue"] and d["avg_total_assets"] and d["avg_equity"]:
            indicators["杜邦分析"] = dupont_analysis(
                net_income=d["net_income"],
                revenue=d["revenue"],
                avg_total_assets=d["avg_total_assets"],
                avg_equity=d["avg_equity"],
            )
        else:
            missing.append("杜邦分析（数据不完整）")
    except Exception as exc:
        missing.append(f"杜邦分析（异常: {exc}）")

    # 2) Altman Z-Score -- 根据行业选用对应版本
    industry = state.get("industry", "")
    try:
        if d["total_assets"] and d["total_liabilities"]:
            if industry == "制造业":
                indicators["Z-Score"] = altman_zscore_manufacturing(
                    working_capital=d["working_capital"],
                    retained_earnings=d["retained_earnings"],
                    ebit=d["ebit"],
                    book_equity=d["equity"],
                    total_liabilities=d["total_liabilities"],
                    total_assets=d["total_assets"],
                    revenue=d["revenue"],
                )
            else:
                indicators["Z-Score"] = altman_zscore_service(
                    working_capital=d["working_capital"],
                    retained_earnings=d["retained_earnings"],
                    ebit=d["ebit"],
                    book_equity=d["equity"],
                    total_liabilities=d["total_liabilities"],
                    total_assets=d["total_assets"],
                )
        else:
            missing.append("Z-Score（数据不完整）")
    except Exception as exc:
        missing.append(f"Z-Score（异常: {exc}）")

    # 3) 偿债能力
    try:
        if d["current_assets"] and d["total_assets"]:
            indicators["偿债能力"] = solvency_ratios(
                current_assets=d["current_assets"],
                current_liabilities=d["current_liabilities"],
                inventory=d["inventory"],
                total_assets=d["total_assets"],
                total_liabilities=d["total_liabilities"],
            )
        else:
            missing.append("偿债能力指标（数据不完整）")
    except Exception as exc:
        missing.append(f"偿债能力（异常: {exc}）")

    # 4) 盈利能力
    try:
        if d["net_income"] and d["revenue"]:
            indicators["盈利能力"] = profitability_ratios(
                net_income=d["net_income"],
                revenue=d["revenue"],
                gross_profit=d["gross_profit"],
                total_assets=d["total_assets"],
                equity=d["equity"],
            )
        else:
            missing.append("盈利能力指标（数据不完整）")
    except Exception as exc:
        missing.append(f"盈利能力（异常: {exc}）")

    # 5) 营运能力
    try:
        if d["revenue"] and d["avg_total_assets"]:
            indicators["营运能力"] = efficiency_ratios(
                revenue=d["revenue"],
                avg_receivables=d["avg_receivables"],
                cost_of_goods=d["cost"],
                avg_inventory=d["avg_inventory"],
                avg_total_assets=d["avg_total_assets"],
            )
        else:
            missing.append("营运能力指标（数据不完整）")
    except Exception as exc:
        missing.append(f"营运能力（异常: {exc}）")

    if missing:
        indicators["计算缺失项"] = missing
        for m in missing:
            logger.warning("指标缺失: %s", m)

    computed = [k for k in indicators if k != "计算缺失项"]
    logger.info("指标计算完成: %s", ", ".join(computed) if computed else "无")

    return {"indicators": indicators, "errors": errors}
