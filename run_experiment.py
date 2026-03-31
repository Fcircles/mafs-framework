"""实验运行主入口 -- 并行框架运行、基线对比、评估与结果输出"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import (
    DATA_DIR, OUTPUT_DIR, INDUSTRIES, YEARS,
    CASE_STUDY_COMPANIES, RAGConfig, ConcurrencyConfig,
)
from models import AnalysisResult
from agents.orchestrator import (
    run_single_report, run_company_reports, run_batch_reports,
    save_results, _find_pdf,
    load_checkpoints, _save_checkpoint, _ckpt_key,
)
from baselines import (
    RuleBasedBaseline, SingleLLMBaseline, GeneralRAGBaseline,
)
from evaluation.metrics import evaluate_batch, cohens_kappa
from utils.llm_client import warmup as _warmup_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "experiment.log", encoding="utf-8"),
    ],
    force=True,
)
for _h in logging.getLogger().handlers:
    _h.flush = _h.stream.flush if hasattr(_h, "stream") else _h.flush
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("run_experiment")

ROBUSTNESS_COMPANIES = {
    "科创板": [
        ("688981", "中芯国际"), ("688111", "金山办公"), ("688363", "华熙生物"),
        ("688036", "传音控股"), ("688008", "澜起科技"),
    ],
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _iter_companies(industries: dict | None = None, years: list[int] | None = None):
    """遍历公司和年份，yield (code, name, industry, year)。"""
    industries = industries or INDUSTRIES
    years = years or YEARS
    for industry, companies in industries.items():
        for code, name in companies:
            for year in years:
                yield code, name, industry, year


def _save_json(data, filename: str, output_dir: Path | None = None) -> Path:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info("已保存: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. 联调测试 (smoke test)
# ---------------------------------------------------------------------------

def _log_report_summary(result: dict, failed_cases: list[str]) -> None:
    """打印单份年报分析结果摘要。"""
    name = result.get("company_name", "?")
    year = result.get("year", "?")
    inds = result.get("indicators", {})
    computed = [k for k in inds if k != "计算缺失项" and isinstance(inds[k], dict)]
    missing = inds.get("计算缺失项", [])
    evidence_count = len(result.get("evidence_map", []))
    interp_len = len(result.get("interpretation", ""))
    errors = result.get("errors", [])

    logger.info("  [%s %s] 指标类别: %s | 证据: %d | 解读: %d字 | 缺失: %s",
                name, year, computed, evidence_count, interp_len, missing or "无")
    if errors:
        logger.warning("  [%s %s] 错误: %s", name, year, errors)
        failed_cases.append(f"{name}_{year}: {errors}")


def run_smoke_test():
    """阶段2.5: 从3个行业各选1份年报 + 2份边界case，全部并行跑通验证。"""
    logger.info("=" * 70)
    logger.info("开始联调测试 (smoke test) -- 并行模式")
    logger.info("=" * 70)

    test_cases = [
        ("002594", "比亚迪", "制造业", 2023),
        ("600519", "贵州茅台", "消费行业", 2023),
        ("600276", "恒瑞医药", "医药行业", 2023),
        ("002594", "比亚迪", "制造业", 2024),
        ("688981", "中芯国际", "科创板", 2023),
    ]

    # ---- 第一步: 多智能体框架端到端验证 (并行) ----
    logger.info(">>> 第1步: 多智能体框架端到端验证 (%d 份年报, 并行)", len(test_cases))

    batch_tasks: list[dict] = []
    for code, name, industry, year in test_cases:
        pdf = _find_pdf(code, name, industry, year)
        batch_tasks.append({
            "pdf_path": str(pdf) if pdf else None,
            "company_code": code,
            "company_name": name,
            "industry": industry,
            "year": year,
        })

    t0 = time.time()
    results = run_batch_reports(batch_tasks)
    elapsed = time.time() - t0
    logger.info("第1步完成: %d 份, 耗时 %.1f 分钟", len(results), elapsed / 60)

    failed_cases: list[str] = []
    for r in results:
        _log_report_summary(r, failed_cases)
    _save_json(results, "smoke_test_results.json")

    # ---- 第二步: 全部3个基线冒烟测试 (并行) ----
    logger.info(">>> 第2步: 基线冒烟测试 (3个基线, 并行)")

    baselines_all = [
        RuleBasedBaseline(),
        SingleLLMBaseline(),
        GeneralRAGBaseline(),
    ]
    smoke_case = test_cases[0]
    code, name, industry, year = smoke_case
    pdf = _find_pdf(code, name, industry, year)
    baseline_results_map: dict[str, list[AnalysisResult]] = {}

    if pdf is not None:
        def _run_baseline(bl):
            return bl.name, bl.analyze(str(pdf), code, name, year, industry)

        max_w = len(baselines_all)
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(_run_baseline, bl): bl for bl in baselines_all}
            for future in as_completed(futures):
                bl = futures[future]
                try:
                    bl_name, bl_result = future.result()
                    baseline_results_map[bl_name] = [bl_result]
                    ind_count = len(bl_result.indicators)
                    aligned = sum(1 for ind in bl_result.indicators
                                  if ind.source_page is not None and (ind.source_text or "").strip())
                    logger.info("  %s: %d 个指标, %d 个有证据, 解读 %d 字",
                                bl_name, ind_count, aligned, len(bl_result.interpretation))
                except Exception as exc:
                    logger.error("  %s 失败: %s", bl.name, exc, exc_info=True)
                    failed_cases.append(f"基线_{bl.name}: {exc}")

    bl_serialized = {
        bname: [r.to_dict() for r in res]
        for bname, res in baseline_results_map.items()
    }
    _save_json(bl_serialized, "smoke_test_baseline_results.json")

    # ---- 第三步: 评价指标验证 ----
    logger.info(">>> 第3步: 评价指标验证")

    if results:
        from evaluation.metrics import (
            evidence_alignment_rate, caliber_consistency, interpretation_quality,
        )

        agent_analysis = [AnalysisResult.from_pipeline_output(r) for r in results]

        for ar in agent_analysis:
            ea = evidence_alignment_rate(ar.indicators)
            logger.info("  证据对齐率 [%s %d]: %.1f%% (%d/%d)",
                        ar.company_name, ar.year,
                        ea["alignment_rate"] * 100,
                        ea["aligned_count"], ea["total_count"])

        byd_group = [ar for ar in agent_analysis
                     if ar.company_code == "002594"]
        if len(byd_group) >= 2:
            cc = caliber_consistency(byd_group)
            logger.info("  口径一致性 [比亚迪 多年]: %.1f%% (%d/%d)",
                        cc["consistency"] * 100,
                        cc["consistent_count"], cc["total_count"])

        test_ar = agent_analysis[0]
        logger.info("  LLM-as-Judge 评分测试 (1次运行): %s %d",
                    test_ar.company_name, test_ar.year)
        try:
            iq = interpretation_quality(
                test_ar.interpretation,
                test_ar.indicators,
                {"company_name": test_ar.company_name,
                 "year": test_ar.year,
                 "industry": test_ar.industry},
                n_runs=1,
            )
            logger.info("    评审A均分: %.2f, 评审B均分: %.2f, 综合: %.2f",
                        iq["reviewer_a"]["mean"],
                        iq["reviewer_b"]["mean"],
                        iq["overall_score"])
        except Exception as exc:
            logger.error("    LLM-as-Judge 评分失败: %s", exc, exc_info=True)
            failed_cases.append(f"LLM-as-Judge: {exc}")

    # ---- 汇总 ----
    logger.info("=" * 70)
    logger.info("联调测试完成")
    logger.info("  框架端到端: %d/%d 成功", len(results), len(test_cases))
    logger.info("  基线冒烟: %d/%d 成功",
                len(baseline_results_map), len(baselines_all))
    if failed_cases:
        logger.warning("  存在问题 (%d 项):", len(failed_cases))
        for fc in failed_cases:
            logger.warning("    - %s", fc)
    else:
        logger.info("  全部通过，无阻塞性错误")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# 2. 主实验: 多智能体框架
# ---------------------------------------------------------------------------

def run_main_experiment(industries: dict | None = None,
                        years: list[int] | None = None,
                        checkpoint_name: str = "main",
                        save_as: str | None = None):
    """并行运行完整多智能体框架，支持断点续传。

    Parameters
    ----------
    save_as : 结果文件名（默认仅当 checkpoint_name=="main" 时保存为
              main_experiment_results.json，避免稳健性检验覆盖主结果）。
    """
    industries = industries or INDUSTRIES
    years = years or YEARS

    logger.info("=" * 70)
    logger.info("开始主实验 (并行): %d 个行业, %d 年", len(industries), len(years))
    logger.info("=" * 70)

    batch_tasks: list[dict] = []
    for industry, companies in industries.items():
        for code, name in companies:
            for year in years:
                pdf = _find_pdf(code, name, industry, year)
                batch_tasks.append({
                    "pdf_path": str(pdf) if pdf else None,
                    "company_code": code,
                    "company_name": name,
                    "industry": industry,
                    "year": year,
                })

    ckpt_dir = OUTPUT_DIR / "checkpoints" / checkpoint_name

    logger.info("共 %d 份年报待分析 (checkpoint: %s)", len(batch_tasks), ckpt_dir)
    t0 = time.time()
    all_results = run_batch_reports(batch_tasks, checkpoint_dir=ckpt_dir)
    elapsed = time.time() - t0

    failed = [
        {"company": r.get("company_name"), "year": r.get("year"), "errors": r["errors"]}
        for r in all_results if r.get("errors")
    ]
    logger.info("主实验完成: %d 份报告, 耗时 %.1f 分钟, %d 份有错误",
                len(all_results), elapsed / 60, len(failed))

    result_filename = save_as or ("main_experiment_results.json" if checkpoint_name == "main" else None)
    if result_filename:
        _save_json(all_results, result_filename)
    if failed and checkpoint_name == "main":
        _save_json(failed, "main_experiment_errors.json")

    return all_results


# ---------------------------------------------------------------------------
# 3. 基线对比
# ---------------------------------------------------------------------------

def _run_single_baseline(bl, pdf_path, code, name, year, industry,
                         max_retries: int = 3):
    """在线程池中运行单个基线分析任务（带任务级重试）。"""
    import time as _time
    for attempt in range(max_retries):
        try:
            return bl.analyze(str(pdf_path), code, name, year, industry)
        except Exception:
            if attempt < max_retries - 1:
                delay = 10 * (attempt + 1)
                logger.warning(
                    "%s 任务级重试 %d/%d (%s %d), %ds后重试",
                    bl.name, attempt + 1, max_retries, name, year, delay,
                )
                _time.sleep(delay)
                continue
            raise


def run_baselines(industries: dict | None = None,
                  years: list[int] | None = None):
    """并行运行3个基线方法，支持断点续传。"""
    industries = industries or INDUSTRIES
    years = years or YEARS

    baselines = [
        RuleBasedBaseline(),
        SingleLLMBaseline(),
        GeneralRAGBaseline(),
    ]

    logger.info("=" * 70)
    logger.info("开始基线对比实验 (并行): %d 个基线", len(baselines))
    logger.info("=" * 70)

    tasks_list: list[tuple] = []
    for code, name, industry, year in _iter_companies(industries, years):
        pdf = _find_pdf(code, name, industry, year)
        if pdf is not None:
            tasks_list.append((pdf, code, name, industry, year))
        else:
            logger.warning("跳过 %s %d (PDF不存在)", name, year)

    for bl in baselines:
        ckpt_dir = OUTPUT_DIR / "checkpoints" / f"baseline_{bl.name}"
        cached = load_checkpoints(ckpt_dir) if ckpt_dir.exists() else {}
        remaining = [
            t for t in tasks_list
            if _ckpt_key(t[1], t[4]) not in cached
        ]
        logger.info("--- 基线: %s (%d 份, 已完成 %d, 剩余 %d) ---",
                    bl.name, len(tasks_list), len(cached), len(remaining))
        if not remaining:
            logger.info("基线 %s 全部已有 checkpoint，跳过", bl.name)
            serialized = list(cached.values())
            _save_json(serialized, f"baseline_{bl.name}_results.json")
            continue

        bl_results: list[AnalysisResult] = []
        t0 = time.time()
        done_count = 0
        total_remaining = len(remaining)

        max_w = ConcurrencyConfig.REPORT_MAX_WORKERS or 30
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            future_map = {
                executor.submit(
                    _run_single_baseline, bl, pdf, code, name, year, industry,
                ): (code, name, year)
                for pdf, code, name, industry, year in remaining
            }
            for future in as_completed(future_map):
                ccode, cname, cyear = future_map[future]
                try:
                    ar = future.result()
                    bl_results.append(ar)
                    _save_checkpoint(ar.to_dict(), ckpt_dir)
                except Exception as exc:
                    logger.error("%s 分析失败 (%s %d): %s",
                                 bl.name, cname, cyear, exc)
                done_count += 1
                if done_count % 5 == 0 or done_count == total_remaining:
                    elapsed_m = (time.time() - t0) / 60
                    logger.info("%s 进度: %d/%d (总含已恢复: %d/%d), 已耗时 %.1f分钟",
                                bl.name, done_count, total_remaining,
                                done_count + len(cached), len(tasks_list), elapsed_m)

        elapsed = time.time() - t0
        logger.info("%s 完成: %d 份, 耗时 %.1f 分钟",
                    bl.name, len(bl_results), elapsed / 60)

        all_serialized = list(cached.values()) + [r.to_dict() for r in bl_results]
        _save_json(all_serialized, f"baseline_{bl.name}_results.json")


# ---------------------------------------------------------------------------
# 4. 评估
# ---------------------------------------------------------------------------

def _build_ground_truths(agent_results: list[AnalysisResult]) -> dict[str, dict[str, float]]:
    """从多智能体框架确定性公式计算结果构建参考基准。

    注意：此基准用于衡量基线方法与确定性公式计算结果的一致程度，
    multi_agent方法自身的准确性通过formula_verification独立验证。
    因此multi_agent的calculation_correctness恒为100%是预期行为。

    只保留任务书中定义的核心指标（排除 Z-Score 中间变量 X1-X5 等），
    确保评价的分母是各方法均应计算的标准指标集。
    """
    from evaluation.metrics import _normalize_indicator_key, CORE_INDICATORS

    gt: dict[str, dict[str, float]] = {}
    for r in agent_results:
        key = f"{r.company_code}_{r.year}"
        indicator_map: dict[str, float] = {}
        for ind in r.indicators:
            if ind.value is not None:
                composite_key = _normalize_indicator_key(ind.category, ind.name)
                if composite_key in CORE_INDICATORS and composite_key not in indicator_map:
                    indicator_map[composite_key] = ind.value
        if indicator_map:
            gt[key] = indicator_map
    return gt


def run_evaluation():
    """加载实验结果并计算评价指标。"""
    logger.info("=" * 70)
    logger.info("开始评估")
    logger.info("=" * 70)

    main_path = OUTPUT_DIR / "main_experiment_results.json"
    if not main_path.exists():
        logger.error("主实验结果不存在: %s", main_path)
        return

    with open(main_path, "r", encoding="utf-8") as f:
        raw_results = json.load(f)

    agent_results = [AnalysisResult.from_pipeline_output(r) for r in raw_results]
    logger.info("加载了 %d 份多智能体框架结果", len(agent_results))

    ground_truths = _build_ground_truths(agent_results)
    logger.info("构建 ground truth: %d 份 (基于多智能体确定性公式计算)", len(ground_truths))

    all_reports: dict[str, dict] = {}

    if agent_results:
        multi_year_groups: dict[str, list[AnalysisResult]] = {}
        for r in agent_results:
            multi_year_groups.setdefault(r.company_code, []).append(r)

        agent_report = evaluate_batch(
            agent_results,
            ground_truths=None,
            multi_year_groups=multi_year_groups,
            run_llm_judge=True,
            n_runs=3,
        )
        agent_report["method"] = "multi_agent"

        from evaluation.metrics import formula_verification
        fv_scores = []
        for r in agent_results:
            fv = formula_verification(r.indicators)
            fv_scores.append(fv["pass_rate"])
        agent_report["formula_verification"] = {
            "mean": float(np.mean(fv_scores)),
            "std": float(np.std(fv_scores)),
            "count": len(fv_scores),
        }

        all_reports["multi_agent"] = agent_report
        logger.info("多智能体框架评估完成")

    baseline_names = ["rule_based", "single_llm", "general_rag"]
    for bl_name in baseline_names:
        bl_path = OUTPUT_DIR / f"baseline_{bl_name}_results.json"
        if not bl_path.exists():
            logger.warning("基线结果不存在: %s", bl_path)
            continue

        with open(bl_path, "r", encoding="utf-8") as f:
            bl_raw = json.load(f)

        bl_results = [AnalysisResult.from_dict(r) for r in bl_raw]
        if not bl_results:
            continue

        multi_year_groups = {}
        for r in bl_results:
            multi_year_groups.setdefault(r.company_code, []).append(r)

        bl_report = evaluate_batch(
            bl_results,
            ground_truths=ground_truths,
            multi_year_groups=multi_year_groups,
            run_llm_judge=True,
            n_runs=3,
        )
        bl_report["method"] = bl_name
        all_reports[bl_name] = bl_report
        logger.info("%s 评估完成", bl_name)

    # ---- 全局 Cohen's Kappa（跨方法汇总，写入报告） ----
    all_a_overalls: list[float] = []
    all_b_overalls: list[float] = []
    for method, report in all_reports.items():
        iq = report.get("interpretation_quality")
        if not iq:
            continue
        raw_a = iq.get("_raw_a_overalls", [])
        raw_b = iq.get("_raw_b_overalls", [])
        all_a_overalls.extend(raw_a)
        all_b_overalls.extend(raw_b)

    if len(all_a_overalls) >= 10:
        global_kappa = cohens_kappa(all_a_overalls, all_b_overalls)
        all_reports["_global"] = {
            "cohens_kappa_cross_method": {
                "value": global_kappa,
                "interpretation": _interpret_kappa_str(global_kappa),
                "sample_count": len(all_a_overalls),
                "note": "跨所有方法的评审A与评审B综合质量评分的加权Kappa系数",
            },
        }
        logger.info("全局跨方法 Cohen's Kappa = %.4f (%s), 样本数=%d",
                     global_kappa, _interpret_kappa_str(global_kappa),
                     len(all_a_overalls))

    for report in all_reports.values():
        iq = report.get("interpretation_quality")
        if iq:
            iq.pop("_raw_a_overalls", None)
            iq.pop("_raw_b_overalls", None)

    _save_json(all_reports, "evaluation_report.json")

    _print_comparison_table(all_reports)

    return all_reports


def _interpret_kappa_str(kappa: float) -> str:
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


def _print_comparison_table(reports: dict[str, dict]):
    """打印对比结果摘要表。"""
    header = f"{'方法':<20} {'计算/公式验证':>12} {'证据对齐率':>10} {'口径一致性':>10} {'解读质量':>10} {'Kappa':>8}"
    logger.info("\n" + "=" * 80)
    logger.info(header)
    logger.info("-" * 80)

    for method, report in reports.items():
        if method.startswith("_"):
            continue
        cc = report.get("calculation_correctness") or {}
        fv = report.get("formula_verification") or {}
        ea = report.get("evidence_alignment") or {}
        cal = report.get("caliber_consistency") or {}
        iq = report.get("interpretation_quality") or {}
        kappa = report.get("cohens_kappa") or {}

        if cc:
            calc_str = f"{cc.get('mean', 0) * 100:>10.1f}%"
        elif fv:
            calc_str = f"{fv.get('mean', 0) * 100:>10.1f}%"
        else:
            calc_str = f"{'N/A':>11}"

        row = (
            f"{method:<20} "
            f"{calc_str} "
            f"{ea.get('mean', 0) * 100:>9.1f}% "
            f"{cal.get('mean', 0) * 100:>9.1f}% "
            f"{iq.get('overall_mean', 0):>9.2f} "
            f"{kappa.get('value', 0):>7.3f}"
        )
        logger.info(row)

    gk = reports.get("_global", {}).get("cohens_kappa_cross_method", {})
    if gk:
        logger.info("-" * 80)
        logger.info("全局跨方法 Kappa = %.4f (%s, n=%d)",
                     gk["value"], gk["interpretation"], gk["sample_count"])
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# 5. 案例分析
# ---------------------------------------------------------------------------

def run_case_study():
    """并行运行3家典型公司的深度案例分析。"""
    logger.info("=" * 70)
    logger.info("开始案例分析 (并行)")
    logger.info("=" * 70)

    max_w = ConcurrencyConfig.REPORT_MAX_WORKERS or 30
    with ThreadPoolExecutor(max_workers=max_w) as executor:
        future_map = {
            executor.submit(run_company_reports, code, name, industry, YEARS): (code, name, industry)
            for code, name, industry in CASE_STUDY_COMPANIES
        }
        for future in as_completed(future_map):
            code, name, industry = future_map[future]
            try:
                results = future.result()
                save_results(results, OUTPUT_DIR / "case_study")
                for r in results:
                    year = r.get("year", "?")
                    inds = r.get("indicators", {})
                    interp = r.get("interpretation", "")
                    logger.info("  [%s] %s年: %d 类指标, 解读 %d 字",
                                name, year,
                                sum(1 for k in inds if k != "计算缺失项"
                                    and isinstance(inds.get(k), dict)),
                                len(interp))
            except Exception as exc:
                logger.error("案例分析失败 (%s): %s", name, exc, exc_info=True)

    logger.info("案例分析完成")


# ---------------------------------------------------------------------------
# 6. 稳健性检验
# ---------------------------------------------------------------------------

def run_robustness():
    """并行运行3组稳健性检验。"""
    logger.info("=" * 70)
    logger.info("开始稳健性检验 (并行)")
    logger.info("=" * 70)

    rob_dir = OUTPUT_DIR / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- 检验1: 更换数据集 (科创板) ---")
    kcb_results = run_main_experiment(
        industries=ROBUSTNESS_COMPANIES, years=YEARS,
        checkpoint_name="robustness_dataset",
    )
    _save_json(kcb_results, "robustness_dataset.json", rob_dir)

    logger.info("--- 检验2: 调整RAG Top-k参数 ---")
    original_top_k = RAGConfig.TOP_K
    for k_val in [3, 10]:
        RAGConfig.TOP_K = k_val
        logger.info("Top-k = %d", k_val)
        topk_results: list[dict] = []
        for test_code, test_name, test_industry in CASE_STUDY_COMPANIES:
            results = run_company_reports(
                test_code, test_name, test_industry, [2022, 2023, 2024],
            )
            topk_results.extend(results)

        for r in topk_results:
            r["topk_setting"] = k_val
        _save_json(topk_results, f"robustness_topk_{k_val}.json", rob_dir)
    RAGConfig.TOP_K = original_top_k

    logger.info("--- 检验3: 调整时间窗口 (2022-2024) ---")
    short_results = run_main_experiment(
        industries=INDUSTRIES, years=[2022, 2023, 2024],
        checkpoint_name="robustness_timewindow",
    )
    _save_json(short_results, "robustness_timewindow.json", rob_dir)

    logger.info("稳健性检验完成")


# ---------------------------------------------------------------------------
# 6.5 稳健性结果评估
# ---------------------------------------------------------------------------

def run_robustness_evaluation():
    """对已有的稳健性结果文件运行评估，生成与主实验同口径的指标报告。

    不运行 LLM-as-Judge（节省 API 开销），仅计算客观指标：
    公式验证通过率、证据对齐率、口径一致性、指标覆盖率。
    """
    from evaluation.metrics import (
        formula_verification, evidence_alignment_rate,
        caliber_consistency as cc_func,
    )

    logger.info("=" * 70)
    logger.info("开始稳健性结果评估")
    logger.info("=" * 70)

    rob_dir = OUTPUT_DIR / "robustness"
    rob_report: dict[str, dict] = {}

    robustness_files = {
        "dataset_科创板": "robustness_dataset.json",
        "topk_3": "robustness_topk_3.json",
        "topk_10": "robustness_topk_10.json",
        "timewindow_2022_2024": "robustness_timewindow.json",
    }

    main_gt_path = OUTPUT_DIR / "main_experiment_results.json"
    ground_truths: dict[str, dict[str, float]] | None = None
    if main_gt_path.exists():
        with open(main_gt_path, "r", encoding="utf-8") as f:
            main_raw = json.load(f)
        main_results = [AnalysisResult.from_pipeline_output(r) for r in main_raw]
        ground_truths = _build_ground_truths(main_results)

    for label, filename in robustness_files.items():
        fpath = rob_dir / filename
        if not fpath.exists():
            logger.warning("稳健性结果不存在: %s", fpath)
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        results = [AnalysisResult.from_pipeline_output(r) for r in raw]
        if not results:
            continue

        fv_scores = [formula_verification(r.indicators)["pass_rate"] for r in results]
        ea_scores = [evidence_alignment_rate(r.indicators)["alignment_rate"] for r in results]

        multi_year_groups: dict[str, list[AnalysisResult]] = {}
        for r in results:
            multi_year_groups.setdefault(r.company_code, []).append(r)
        cc_scores = [cc_func(group)["consistency"]
                     for group in multi_year_groups.values()
                     if len(group) >= 2]

        by_cat: dict[str, dict[str, int]] = {}
        expected_cats = ["杜邦分析", "偿债能力", "盈利能力", "营运能力", "Z-Score"]
        for r in results:
            cats_present = {ind.category for ind in r.indicators if ind.value is not None}
            for cat in expected_cats:
                by_cat.setdefault(cat, {"count": 0, "total": 0})
                by_cat[cat]["total"] += 1
                if cat in cats_present:
                    by_cat[cat]["count"] += 1

        ic_overall = sum(d["count"] for d in by_cat.values()) / max(sum(d["total"] for d in by_cat.values()), 1)

        entry: dict = {
            "label": label,
            "sample_count": len(results),
            "formula_verification": {
                "mean": float(np.mean(fv_scores)),
                "std": float(np.std(fv_scores)),
            },
            "evidence_alignment": {
                "mean": float(np.mean(ea_scores)),
                "std": float(np.std(ea_scores)),
            },
            "indicator_coverage": {
                "per_category": {cat: {"rate": d["count"] / max(d["total"], 1)}
                                 for cat, d in by_cat.items()},
                "overall": ic_overall,
            },
        }
        if cc_scores:
            entry["caliber_consistency"] = {
                "mean": float(np.mean(cc_scores)),
                "std": float(np.std(cc_scores)),
                "company_count": len(cc_scores),
            }

        rob_report[label] = entry
        logger.info("  %s: n=%d, 公式验证=%.3f, 证据对齐=%.3f, 指标覆盖=%.3f",
                     label, len(results),
                     entry["formula_verification"]["mean"],
                     entry["evidence_alignment"]["mean"],
                     ic_overall)

    _save_json(rob_report, "robustness_evaluation.json", rob_dir)
    logger.info("稳健性评估完成，结果保存至 robustness/robustness_evaluation.json")
    return rob_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="毕业论文实验运行器")
    parser.add_argument(
        "phase",
        choices=[
            "smoke_test", "main", "baselines", "evaluate",
            "case_study", "robustness", "robustness_eval", "all",
        ],
        help="运行阶段: smoke_test(联调) / main(主实验) / baselines(基线) / "
             "evaluate(评估) / case_study(案例) / robustness(稳健性) / "
             "robustness_eval(稳健性评估) / all(全部)",
    )
    args = parser.parse_args()

    logger.info("预热 LLM 客户端（避免多线程导入死锁）…")
    _warmup_clients()
    logger.info("客户端预热完成")

    phase = args.phase

    if phase == "smoke_test":
        run_smoke_test()
    elif phase == "main":
        run_main_experiment()
    elif phase == "baselines":
        run_baselines()
    elif phase == "evaluate":
        run_evaluation()
    elif phase == "case_study":
        run_case_study()
    elif phase == "robustness":
        run_robustness()
    elif phase == "robustness_eval":
        run_robustness_evaluation()
    elif phase == "all":
        run_main_experiment()
        run_baselines()
        run_evaluation()
        run_case_study()
        run_robustness()
        run_robustness_evaluation()


if __name__ == "__main__":
    main()
