"""证据对齐Agent -- 将计算结果关联至年报原文（并发版）"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import PipelineState
from config import ConcurrencyConfig, RAGConfig
from utils.llm_client import get_client_for_provider, get_model_for_provider, chat_completion
from utils.vector_store import VectorStore

logger = logging.getLogger(__name__)
_RELEVANCE_RETRIES = 2

INDICATOR_QUERIES: dict[str, dict[str, str]] = {
    "杜邦分析": {
        "ROE": "净资产收益率 ROE 净利润 股东权益",
        "销售净利率": "销售净利率 净利润 营业收入",
        "资产周转率": "总资产周转率 营业收入 总资产",
        "权益乘数": "权益乘数 总资产 股东权益 杠杆",
    },
    "Z-Score": {
        "Z''-Score": "Altman Z-Score 财务风险 破产预警 营运资本",
        "zone": "财务安全 风险区间 偿债风险",
    },
    "偿债能力": {
        "流动比率": "流动比率 流动资产 流动负债 短期偿债",
        "速动比率": "速动比率 速动资产 流动负债 存货",
        "资产负债率": "资产负债率 总负债 总资产 杠杆率",
    },
    "盈利能力": {
        "毛利率": "毛利率 营业收入 营业成本 毛利",
        "净利率": "净利率 净利润 营业收入 盈利",
        "ROA": "总资产收益率 ROA 净利润 总资产",
        "ROE": "净资产收益率 ROE 净利润 股东权益",
    },
    "营运能力": {
        "应收账款周转率": "应收账款周转率 应收账款 营业收入 回款",
        "存货周转率": "存货周转率 存货 营业成本 周转天数",
        "总资产周转率": "总资产周转率 营业收入 总资产 资产效率",
    },
}


_RELEVANCE_PROMPT = """\
判断以下年报文本片段是否与财务指标"{indicator}"直接相关（包含该指标的原始数据、计算依据或直接说明）。

文本片段：
{text}

仅回答JSON：{{"relevant": true}} 或 {{"relevant": false}}"""


def _llm_relevance_check_single(
    cand: dict,
    indicator_name: str,
    client,
    model: str = "",
) -> dict | None:
    """对单个候选证据进行 LLM 相关性验证。返回通过的候选或 None。"""
    _model = model or get_model_for_provider()
    prompt = _RELEVANCE_PROMPT.format(
        indicator=indicator_name,
        text=cand["source_text"][:300],
    )
    for attempt in range(1, _RELEVANCE_RETRIES + 1):
        try:
            raw = chat_completion(
                client, _model,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = raw.strip()
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                parsed = json.loads(text[brace_start:brace_end + 1])
                if parsed.get("relevant", False):
                    return cand
            return None
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("相关性验证解析失败 (%s, 第%d次): %s",
                         indicator_name, attempt, exc)
    return None


def _align_single_indicator(
    category: str,
    ind_name: str,
    ind_value,
    vs: VectorStore,
    client,
    model: str = "",
) -> list[dict]:
    """为单个指标执行证据检索+LLM过滤，返回证据列表。"""
    query_map = INDICATOR_QUERIES.get(category, {})
    query = query_map.get(ind_name, f"{ind_name} {category}")

    try:
        hits = vs.search(query, top_k=RAGConfig.TOP_K)
    except Exception as exc:
        logger.warning("检索失败 %s/%s: %s", category, ind_name, exc)
        return []

    raw_evidence: list[dict] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        raw_evidence.append({
            "category": category,
            "indicator_name": ind_name,
            "indicator_value": ind_value,
            "source_text": hit["text"][:500],
            "page_number": meta.get("page_number"),
            "section_title": (meta.get("section_titles") or [""])[0]
                if isinstance(meta.get("section_titles"), list)
                else meta.get("section_titles", ""),
            "relevance_score": hit.get("score", 0.0),
        })

    if not raw_evidence:
        return []

    verified: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(raw_evidence)) as pool:
        futures = {
            pool.submit(_llm_relevance_check_single, cand, ind_name, client, model): cand
            for cand in raw_evidence
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                verified.append(result)

    if not verified and raw_evidence:
        logger.debug("LLM过滤后 %s 无有效证据，保留向量相似度最高的结果",
                     ind_name)
        verified.append(raw_evidence[0])

    return verified


def evidence_aligner_node(state: PipelineState) -> dict:
    """证据对齐节点：并发为每个指标检索年报原文中的支持证据。"""

    indicators: dict = state.get("indicators", {})
    vs: VectorStore | None = state.get("vector_store")
    errors: list[str] = list(state.get("errors", []))

    if vs is None or not indicators:
        msg = "向量索引或指标数据缺失，跳过证据对齐"
        logger.warning(msg)
        errors.append(msg)
        return {"evidence_map": [], "errors": errors}

    _provider = state.get("llm_provider", "tengri")
    client = get_client_for_provider(_provider)
    _model = state.get("llm_model") or get_model_for_provider(_provider)

    tasks: list[tuple[str, str, object]] = []
    for category, category_data in indicators.items():
        if category == "计算缺失项" or not isinstance(category_data, dict):
            continue
        for ind_name, ind_value in category_data.items():
            tasks.append((category, ind_name, ind_value))

    evidence_map: list[dict] = []

    if not tasks:
        logger.warning("无指标需要对齐证据，跳过")
        return {"evidence_map": [], "errors": errors}

    max_workers = min(ConcurrencyConfig.EVIDENCE_MAX_WORKERS, len(tasks))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _align_single_indicator, cat, name, val, vs, client, _model,
            ): (cat, name)
            for cat, name, val in tasks
        }
        for future in as_completed(futures):
            cat, name = futures[future]
            try:
                result = future.result()
                evidence_map.extend(result)
            except Exception as exc:
                msg = f"证据对齐异常 ({cat}/{name}): {exc}"
                logger.error(msg)
                errors.append(msg)

    aligned_indicators = len({e["indicator_name"] for e in evidence_map})
    logger.info("证据对齐完成: %d 个指标, %d 条证据 (并发=%d)",
                 aligned_indicators, len(evidence_map), max_workers)

    return {"evidence_map": evidence_map, "errors": errors}
