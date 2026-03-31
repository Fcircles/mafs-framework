"""财务指标计算公式库"""

import pandas as pd


def dupont_analysis(net_income: float, revenue: float,
                    avg_total_assets: float, avg_equity: float) -> dict:
    """
    杜邦分析体系分解
    ROE = 销售净利率 x 资产周转率 x 权益乘数
    """
    net_profit_margin = net_income / revenue if revenue != 0 else 0
    asset_turnover = revenue / avg_total_assets if avg_total_assets != 0 else 0
    equity_multiplier = avg_total_assets / avg_equity if avg_equity != 0 else 0
    roe = net_profit_margin * asset_turnover * equity_multiplier

    return {
        "ROE": roe,
        "销售净利率": net_profit_margin,
        "资产周转率": asset_turnover,
        "权益乘数": equity_multiplier,
    }


def altman_zscore_manufacturing(working_capital: float, retained_earnings: float,
                                 ebit: float, book_equity: float,
                                 total_liabilities: float, total_assets: float,
                                 revenue: float) -> dict:
    """
    Altman Z-Score 原始模型（适用于制造业）
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X4 说明：Altman 原始模型中 X4 = 股东权益市值/总负债。本实验因 A 股年报
    不直接披露市值数据，采用账面所有者权益合计作为替代变量（与 Z'' 修正模型
    对非制造业使用账面权益保持一致），属 A 股场景下的常见简化处理。
    """
    if total_assets == 0:
        return {"Z-Score": None, "zone": "数据缺失"}

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities if total_liabilities != 0 else 0
    x5 = revenue / total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    if z > 2.99:
        zone = "安全区"
    elif z > 1.81:
        zone = "灰色区"
    else:
        zone = "危险区"

    caveat = ""
    if x1 < -0.05:
        caveat = ("注意：该企业营运资本为负（流动负债大于流动资产），"
                  "Z-Score模型可能高估其破产风险。负营运资本在供应链话语权强、"
                  "存货周转快的大型制造企业中较为常见，需结合盈利能力和"
                  "现金流状况综合判断。")

    result = {
        "Z-Score": z,
        "X1_营运资本/总资产": x1,
        "X2_留存收益/总资产": x2,
        "X3_EBIT/总资产": x3,
        "X4_账面权益/总负债": x4,
        "X5_营业收入/总资产": x5,
        "zone": zone,
    }
    if caveat:
        result["model_caveat"] = caveat
    return result


def altman_zscore_service(working_capital: float, retained_earnings: float,
                           ebit: float, book_equity: float,
                           total_liabilities: float, total_assets: float) -> dict:
    """
    Altman Z''-Score 修正模型（适用于消费/医药服务类企业）
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
    """
    if total_assets == 0:
        return {"Z''-Score": None, "zone": "数据缺失"}

    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities if total_liabilities != 0 else 0

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    if z > 2.60:
        zone = "安全区"
    elif z > 1.10:
        zone = "灰色区"
    else:
        zone = "危险区"

    caveat = ""
    if x1 < -0.05:
        caveat = ("注意：该企业营运资本为负（流动负债大于流动资产），"
                  "Z''-Score模型中X1权重高达6.56，可能严重高估其破产风险。"
                  "需结合盈利能力和现金流状况综合判断。")

    result = {
        "Z''-Score": z,
        "X1_营运资本/总资产": x1,
        "X2_留存收益/总资产": x2,
        "X3_EBIT/总资产": x3,
        "X4_账面权益/总负债": x4,
        "zone": zone,
    }
    if caveat:
        result["model_caveat"] = caveat
    return result


def solvency_ratios(current_assets: float, current_liabilities: float,
                     inventory: float, total_assets: float,
                     total_liabilities: float) -> dict:
    """偿债能力指标"""
    return {
        "流动比率": current_assets / current_liabilities if current_liabilities != 0 else 0,
        "速动比率": (current_assets - inventory) / current_liabilities if current_liabilities != 0 else 0,
        "资产负债率": total_liabilities / total_assets if total_assets != 0 else 0,
    }


def profitability_ratios(net_income: float, revenue: float,
                          gross_profit: float, total_assets: float,
                          equity: float) -> dict:
    """盈利能力指标"""
    return {
        "毛利率": gross_profit / revenue if revenue != 0 else 0,
        "净利率": net_income / revenue if revenue != 0 else 0,
        "ROA": net_income / total_assets if total_assets != 0 else 0,
        "ROE": net_income / equity if equity != 0 else 0,
    }


def efficiency_ratios(revenue: float, avg_receivables: float,
                       cost_of_goods: float, avg_inventory: float,
                       avg_total_assets: float) -> dict:
    """营运能力指标"""
    return {
        "应收账款周转率": revenue / avg_receivables if avg_receivables != 0 else 0,
        "存货周转率": cost_of_goods / avg_inventory if avg_inventory != 0 else 0,
        "总资产周转率": revenue / avg_total_assets if avg_total_assets != 0 else 0,
    }
