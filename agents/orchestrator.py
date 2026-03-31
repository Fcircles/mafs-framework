"""多Agent协作编排器 -- 基于LangGraph实现Agent调度

两阶段并行策略：
  阶段A — 多进程预解析PDF（绕过GIL，充分利用M4 Pro多核）
  阶段B — 多线程并发跑API密集的LangGraph流水线

支持断点续传：每完成一份年报立即写入 checkpoint，重启时自动跳过。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from langgraph.graph import START, END, StateGraph

from agents import PipelineState
from agents.document_parser import document_parser_node
from agents.indicator_calculator import indicator_calculator_node
from agents.evidence_aligner import evidence_aligner_node
from agents.interpretation_generator import interpretation_generator_node
from agents.consistency_checker import (
    consistency_checker_node,
    check_cross_year_consistency,
    MAX_CONSISTENCY_RETRIES,
)
from config import DATA_DIR, OUTPUT_DIR, INDUSTRIES, ConcurrencyConfig

logger = logging.getLogger(__name__)

_PDF_PARSE_WORKERS = int(os.getenv("PDF_PARSE_WORKERS", "8"))
_ckpt_lock = threading.Lock()

# ---------------------------------------------------------------------------
# LangGraph 状态图构建
# ---------------------------------------------------------------------------

def _should_retry(state: PipelineState) -> str:
    """条件边：判断一致性检验后是结束、还是回退重试解读生成。"""
    if state.get("consistency_passed", True):
        return "end"
    indicators = state.get("indicators", {})
    has_data = any(isinstance(v, dict) for v in indicators.values())
    if not has_data:
        logger.warning("指标数据缺失，无法通过重试修复，直接结束")
        return "end"
    if state.get("retry_count", 0) > MAX_CONSISTENCY_RETRIES:
        logger.warning("一致性检验未通过但已达重试上限，强制结束")
        return "end"
    logger.info("一致性检验未通过，回退重新生成解读 (retry_count=%d)",
                state.get("retry_count", 0))
    return "retry"


def _build_workflow() -> StateGraph:
    workflow = StateGraph(PipelineState)

    workflow.add_node("document_parser", document_parser_node)
    workflow.add_node("indicator_calculator", indicator_calculator_node)
    workflow.add_node("evidence_aligner", evidence_aligner_node)
    workflow.add_node("interpretation_generator", interpretation_generator_node)
    workflow.add_node("consistency_checker", consistency_checker_node)

    workflow.add_edge(START, "document_parser")
    workflow.add_edge("document_parser", "indicator_calculator")
    workflow.add_edge("indicator_calculator", "evidence_aligner")
    workflow.add_edge("evidence_aligner", "interpretation_generator")
    workflow.add_edge("interpretation_generator", "consistency_checker")

    workflow.add_conditional_edges(
        "consistency_checker",
        _should_retry,
        {"end": END, "retry": "interpretation_generator"},
    )

    return workflow


_compiled_app = None


def get_app():
    """获取编译后的 LangGraph 应用（惰性初始化，单例）。"""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = _build_workflow().compile()
    return _compiled_app


# ---------------------------------------------------------------------------
# 结果序列化
# ---------------------------------------------------------------------------

def _serialize_result(state: dict) -> dict:
    """将流水线输出转为可 JSON 序列化的字典，过滤不可序列化对象。"""
    serializable = {}

    for key in ("pdf_path", "company_code", "company_name", "industry", "year",
                "indicators", "evidence_map", "interpretation",
                "consistency_report", "errors"):
        val = state.get(key)
        if val is not None:
            serializable[key] = val

    ev = state.get("extracted_values")
    if ev:
        serializable["extracted_values"] = {
            k: v for k, v in ev.items() if v is not None
        }

    return serializable


# ---------------------------------------------------------------------------
# 断点续传 (checkpoint)
# ---------------------------------------------------------------------------

def _ckpt_key(company_code: str, year: int) -> str:
    return f"{company_code}_{year}"


def load_checkpoints(checkpoint_dir: Path) -> dict[str, dict]:
    """加载已有 checkpoint 结果。返回 {key: result_dict}。"""
    loaded: dict[str, dict] = {}
    if not checkpoint_dir.exists():
        return loaded
    for f in checkpoint_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            key = _ckpt_key(data.get("company_code", ""), data.get("year", 0))
            loaded[key] = data
        except Exception as exc:
            logger.warning("加载 checkpoint 失败 %s: %s", f.name, exc)
    return loaded


def _save_checkpoint(result: dict, checkpoint_dir: Path) -> None:
    """将单份年报结果写入 checkpoint 文件（线程安全）。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    key = _ckpt_key(result.get("company_code", ""), result.get("year", 0))
    path = checkpoint_dir / f"{key}.json"
    with _ckpt_lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 便捷运行函数
# ---------------------------------------------------------------------------

def run_single_report(
    pdf_path: str | Path,
    company_code: str,
    company_name: str,
    industry: str,
    year: int,
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> dict:
    """运行单份年报的完整分析流水线。

    返回可序列化的结果字典。
    """
    pdf_path = str(pdf_path)
    logger.info("=" * 60)
    logger.info("开始分析: %s %s年 (%s) [LLM=%s]", company_name, year, pdf_path, llm_provider)
    logger.info("=" * 60)

    initial_state: PipelineState = {
        "pdf_path": pdf_path,
        "company_code": company_code,
        "company_name": company_name,
        "industry": industry,
        "year": year,
        "retry_count": 0,
        "errors": [],
        "llm_provider": llm_provider,
        "llm_model": llm_model,
    }

    app = get_app()
    final_state = app.invoke(initial_state)

    result = _serialize_result(final_state)
    result["timestamp"] = datetime.now().isoformat()

    logger.info("分析完成: %s %s年", company_name, year)
    return result


def _find_pdf(company_code: str, company_name: str,
              industry: str, year: int,
              data_dir: Path | None = None) -> Path | None:
    """根据公司信息和年份定位年报 PDF 文件。"""
    base = data_dir or DATA_DIR
    filename = f"{company_code}_{company_name}_{year}_年度报告.pdf"

    industry_dirs = {
        "制造业": "制造业",
        "消费行业": "消费行业",
        "医药行业": "医药行业",
        "科创板": "科创板_稳健性",
    }
    sub = industry_dirs.get(industry, industry)
    candidate = base / sub / filename
    if candidate.exists():
        return candidate

    for d in base.iterdir():
        if d.is_dir():
            p = d / filename
            if p.exists():
                return p
    return None


def run_company_reports(
    company_code: str,
    company_name: str,
    industry: str,
    years: list[int],
    data_dir: Path | str | None = None,
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> list[dict]:
    """并行运行单个公司多年年报的分析，并在最后执行跨年一致性检查。

    各年份的分析任务并行提交，完成后按年份排序汇总，
    最后一项的 ``cross_year_consistency`` 键包含跨年一致性检查报告。
    """
    base = Path(data_dir) if data_dir else DATA_DIR

    tasks: list[tuple[int, Path | None]] = []
    for year in sorted(years):
        pdf = _find_pdf(company_code, company_name, industry, year, base)
        tasks.append((year, pdf))

    max_w = ConcurrencyConfig.REPORT_MAX_WORKERS or 30
    results_by_year: dict[int, dict] = {}

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        future_map = {}
        for year, pdf in tasks:
            if pdf is None:
                results_by_year[year] = {
                    "company_code": company_code,
                    "company_name": company_name,
                    "industry": industry,
                    "year": year,
                    "errors": ["PDF文件不存在"],
                }
                continue
            future = executor.submit(
                run_single_report, pdf, company_code, company_name, industry, year,
                llm_provider=llm_provider, llm_model=llm_model,
            )
            future_map[future] = year

        for future in as_completed(future_map):
            year = future_map[future]
            try:
                results_by_year[year] = future.result()
            except Exception as exc:
                logger.error("分析异常 (%s %d): %s", company_name, year, exc)
                results_by_year[year] = {
                    "company_code": company_code,
                    "company_name": company_name,
                    "industry": industry,
                    "year": year,
                    "errors": [str(exc)],
                }

    all_results = [results_by_year[y] for y, _ in tasks if y in results_by_year]

    if len(all_results) >= 2:
        cross_year = check_cross_year_consistency(all_results)
        logger.info("跨年一致性检查: %s", cross_year.get("summary", ""))
        if all_results:
            all_results[-1]["cross_year_consistency"] = cross_year

    return all_results


def _preparse_pdf(pdf_path: str) -> tuple[str, object | None, str | None]:
    """在子进程中解析单个 PDF（绕过 GIL）。返回 (pdf_path, ParsedReport, error)。"""
    try:
        from utils.pdf_parser import parse_annual_report
        report = parse_annual_report(pdf_path)
        return pdf_path, report, None
    except Exception as exc:
        return pdf_path, None, str(exc)


_preparse_cache: dict[str, object] = {}


def run_batch_reports(
    report_tasks: list[dict],
    checkpoint_dir: Path | str | None = None,
) -> list[dict]:
    """两阶段并行运行一批年报分析，支持断点续传。

    阶段A: 多进程预解析所有PDF（绕过GIL，利用M4 Pro多核并行）
    阶段B: 多线程并发跑API密集的LangGraph流水线

    checkpoint_dir 不为 None 时启用断点续传：
      - 已有 checkpoint 的报告直接跳过
      - 每完成一份立即写入 checkpoint
    """
    global _preparse_cache

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
    cached_results: dict[str, dict] = {}
    if ckpt_dir is not None:
        cached_results = load_checkpoints(ckpt_dir)
        if cached_results:
            logger.info("断点续传: 已有 %d 份 checkpoint，将跳过", len(cached_results))

    valid_tasks = [(idx, t) for idx, t in enumerate(report_tasks) if t.get("pdf_path")]
    results: list[dict | None] = [None] * len(report_tasks)

    skipped = 0
    for idx, task in enumerate(report_tasks):
        if task.get("pdf_path") is None:
            results[idx] = {
                "company_code": task.get("company_code", ""),
                "company_name": task.get("company_name", ""),
                "industry": task.get("industry", ""),
                "year": task.get("year", 0),
                "errors": ["PDF文件不存在"],
            }
            continue
        key = _ckpt_key(task.get("company_code", ""), task.get("year", 0))
        if key in cached_results:
            results[idx] = cached_results[key]
            skipped += 1

    remaining_tasks = [
        (idx, t) for idx, t in valid_tasks
        if _ckpt_key(t.get("company_code", ""), t.get("year", 0)) not in cached_results
    ]

    if skipped:
        logger.info("跳过已完成: %d 份, 剩余待分析: %d 份", skipped, len(remaining_tasks))

    if not remaining_tasks:
        logger.info("所有报告均已有 checkpoint，无需重新分析")
        return [r for r in results if r is not None]

    # ---- 阶段A: 多进程预解析PDF ----
    pdfs_to_parse = [t["pdf_path"] for _, t in remaining_tasks
                     if t["pdf_path"] not in _preparse_cache]
    if pdfs_to_parse:
        n_workers = min(_PDF_PARSE_WORKERS, len(pdfs_to_parse))
        logger.info("阶段A: 多进程预解析 %d 份PDF (workers=%d)", len(pdfs_to_parse), n_workers)
        import time
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_preparse_pdf, p): p for p in pdfs_to_parse}
            done_count = 0
            for future in as_completed(futures):
                pdf_path, report, error = future.result()
                done_count += 1
                if report is not None:
                    _preparse_cache[pdf_path] = report
                    if done_count % 10 == 0 or done_count == len(pdfs_to_parse):
                        logger.info("预解析进度: %d/%d", done_count, len(pdfs_to_parse))
                else:
                    logger.warning("预解析失败 %s: %s", Path(pdf_path).name, error)

        logger.info("阶段A完成: %d/%d 成功, 耗时 %.1f秒",
                     len(_preparse_cache), len(pdfs_to_parse), time.time() - t0)

    from agents.document_parser import set_preparse_cache
    set_preparse_cache(_preparse_cache)

    # ---- 阶段B: 多线程并发跑API流水线 ----
    max_w = ConcurrencyConfig.REPORT_MAX_WORKERS or 30
    logger.info("阶段B: 并行分析 %d 份年报 (max_workers=%d)", len(remaining_tasks), max_w)

    done_count = 0
    total_remaining = len(remaining_tasks)

    with ThreadPoolExecutor(max_workers=max_w) as executor:
        future_map = {}
        for idx, task in remaining_tasks:
            future = executor.submit(
                run_single_report,
                task["pdf_path"], task["company_code"], task["company_name"],
                task["industry"], task["year"],
                llm_provider=task.get("llm_provider", "tengri"),
                llm_model=task.get("llm_model", ""),
            )
            future_map[future] = idx

        for future in as_completed(future_map):
            idx = future_map[future]
            task = report_tasks[idx]
            try:
                result = future.result()
                results[idx] = result
                if ckpt_dir is not None:
                    _save_checkpoint(result, ckpt_dir)
            except Exception as exc:
                logger.error("分析异常 (%s %d): %s",
                             task.get("company_name"), task.get("year"), exc)
                results[idx] = {
                    "company_code": task.get("company_code", ""),
                    "company_name": task.get("company_name", ""),
                    "industry": task.get("industry", ""),
                    "year": task.get("year", 0),
                    "errors": [str(exc)],
                }
            done_count += 1
            if done_count % 5 == 0 or done_count == total_remaining:
                logger.info("阶段B进度: %d/%d (总计含已恢复: %d/%d)",
                            done_count, total_remaining,
                            done_count + skipped, len(report_tasks))

    return [r for r in results if r is not None]


def save_results(results: list[dict] | dict, output_dir: Path | str | None = None) -> Path:
    """将分析结果保存为 JSON 文件。

    返回保存的文件路径。
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        results = [results]

    if results:
        first = results[0]
        name = first.get("company_name", "unknown")
        code = first.get("company_code", "000000")
        fname = f"{code}_{name}_分析结果.json"
    else:
        fname = "分析结果.json"

    path = out / fname
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info("结果已保存: %s", path)
    return path
