"""文档解析Agent -- LLM辅助页面识别 + 正则精确提取 + 会计恒等式校验"""

from __future__ import annotations

import json
import re
import logging
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from agents import PipelineState, ITEM_ALIASES, ITEM_FALLBACK_ALIASES, parse_number
from config import ConcurrencyConfig
from utils.llm_client import get_client_for_provider, get_model_for_provider, chat_completion
from utils.pdf_parser import (
    parse_annual_report,
    extract_financial_data,
    get_pages_text_with_metadata,
    ParsedReport,
)
from utils.vector_store import TextChunker, VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 预解析缓存（由 orchestrator 的多进程阶段填充，避免 GIL 瓶颈）
# ---------------------------------------------------------------------------

_preparse_cache: dict[str, ParsedReport] = {}


def set_preparse_cache(cache: dict[str, object]) -> None:
    global _preparse_cache
    _preparse_cache = cache  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 正则常量
# ---------------------------------------------------------------------------

_NUMBER_TOKEN_RE = re.compile(
    r"\(\s*([\d,]+\.?\d*)\s*\)"
    r"|"
    r"(?<![.\d])[-\u2212\u2013\u2014]\s*([\d,]+\.?\d*)"
    r"|"
    r"([\d,]+\.?\d*)"
)

_ITEM_NAME_RE = re.compile(
    r"^([^\d(\uff08\-\u2212\u2013\u2014]*"
    r"(?:[(\uff08][\u4e00-\u9fffa-zA-Z]+[)\uff09][^\d(\uff08\-\u2212\u2013\u2014]*)*)"
    r"\s*(?:\d|[(\uff08][\d\s,]|[\-\u2212\u2013\u2014]\s*\d|$)"
)

_PREFIX_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+\s*[、，,．.]\s*"
    r"|减\s*[:：∶]\s*"
    r"|加\s*[:：∶]\s*"
    r"|其中\s*[:：∶]\s*"
    r")"
)

_FORMAT_NOTE_RE = re.compile(r"[（(][^）)]*(?:填列|以.*号)[^）)]*[）)]?")
_TRAILING_NOTE_RE = re.compile(r"\s+[一二三四五六七八九十]+、.*$")
_NOTE_ANNOTATION_RE = re.compile(r"[（(]\s*注\s*$")
_PAREN_CJK_NOTE_RE = re.compile(
    r"\s*[（(][一二三四五六七八九十\d]{1,4}[）)]\s*")

_COMMA_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+")

_NOTE_REF_SUFFIX_RE = re.compile(
    r"\s+[一二三四五六七八九十]{1,4}(?:\s*[（(][^）)]*[）)])?\s*[,，]?\s*$"
)


def _extract_item_name(line: str) -> str:
    m = _ITEM_NAME_RE.match(line)
    raw = m.group(1).strip() if m else line.strip()
    cleaned = raw
    for _ in range(5):
        prev = cleaned
        cleaned = _PREFIX_RE.sub("", cleaned).strip()
        if cleaned == prev:
            break
    cleaned = _FORMAT_NOTE_RE.sub("", cleaned).strip()
    cleaned = _PAREN_CJK_NOTE_RE.sub("", cleaned).strip()
    cleaned = _TRAILING_NOTE_RE.sub("", cleaned).strip()
    cleaned = _NOTE_REF_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = _NOTE_ANNOTATION_RE.sub("", cleaned).strip()
    return cleaned


def _count_data_lines(text: str) -> int:
    return sum(1 for ln in text.split("\n") if _COMMA_NUMBER_RE.search(ln))


# ---------------------------------------------------------------------------
# 第一步：LLM 辅助识别合并报表页码和单位
# ---------------------------------------------------------------------------

_PAGE_CLASSIFY_PROMPT = """\
你是财务报表结构识别专家。以下是一份年报中若干候选页面的摘要文本。
请识别每个页面的类型和报表单位。

候选页面：
%s

请返回JSON（仅JSON，无其他内容）：
{"pages":[{"page":页码,"type":"合并资产负债表/合并利润表/母公司资产负债表/母公司利润表/其他","unit":"元/千元/万元/未知"}],"consolidated_unit":"元/千元/万元"}"""


def _find_financial_section_start(report: ParsedReport) -> int:
    """定位财务报表章节起始页。

    策略：先找"二、财务报表"标题页，再向前搜索实际报表首页
    （有些年报中报表正文出现在目录标题之前）。
    """
    title_page = None
    for page in report.pages:
        text = page.text or ""
        if re.search(r"[二2][、.]\s*财务报表", text):
            title_page = page.page_number
            break
        if re.search(r"第\s*[一二三四五六七八九十\d]{1,4}\s*[节章]\s*财务", text):
            title_page = page.page_number
            break

    first_table_page = None
    consolidated_re = re.compile(
        r"合\s*并\s*(?:及\s*公\s*司\s*)?资\s*产\s*负\s*债\s*表"
    )
    any_bs_re = re.compile(r"资\s*产\s*负\s*债\s*表")
    for page in report.pages:
        text = page.text or ""
        head = "\n".join(text.split("\n")[:15])
        if consolidated_re.search(head) and _count_data_lines(text) >= 3:
            first_table_page = page.page_number
            break
    if first_table_page is None:
        for page in report.pages:
            text = page.text or ""
            head = "\n".join(text.split("\n")[:15])
            if any_bs_re.search(head) and _count_data_lines(text) >= 5:
                first_table_page = page.page_number
                break

    if first_table_page is None:
        for page in report.pages:
            text = page.text or ""
            bs_kw = sum(1 for kw in ["资产总计", "流动资产合计", "负债合计",
                                      "所有者权益合计"] if kw in text)
            if bs_kw >= 2 and _count_data_lines(text) >= 3:
                first_table_page = page.page_number
                break

    if title_page and first_table_page:
        return min(title_page, first_table_page)
    if first_table_page:
        return first_table_page
    if title_page:
        return title_page
    return max(1, len(report.pages) // 2)


_ANY_FINANCIAL_RE = re.compile(
    r"(?:合\s*并\s*(?:及\s*公\s*司\s*)?|母\s*公\s*司\s*)?"
    r"(?:资\s*产\s*负\s*债\s*表"
    r"|利\s*润\s*(?:及\s*其\s*他\s*综\s*合\s*收\s*益\s*)?表"
    r"|现\s*金\s*流\s*量\s*表)"
)


def _get_candidate_pages(report: ParsedReport) -> list[dict]:
    """收集财务报表章节中含有充足数据行的候选页面。

    两轮搜索：
    1) 前 15 行含报表标题 + 数据行 >= 3 的页面
    2) 无标题但关键词丰富（资产总计/营业收入等 >= 2）的数据页，
       避免标题在上一页底部时整页数据被漏掉
    """
    section_start = _find_financial_section_start(report)
    candidates = []
    seen: set[int] = set()

    for page in report.pages:
        if page.page_number < section_start:
            continue
        text = page.text or ""
        if not text:
            continue
        data_lines = _count_data_lines(text)
        head = "\n".join(text.split("\n")[:15])
        if _ANY_FINANCIAL_RE.search(head) and data_lines >= 3:
            candidates.append({
                "page_number": page.page_number,
                "head": head.strip(),
                "data_lines": data_lines,
            })
            seen.add(page.page_number)

    for page in report.pages:
        if page.page_number < section_start or page.page_number in seen:
            continue
        text = page.text or ""
        dl = _count_data_lines(text)
        if dl < 5:
            continue
        head = "\n".join(text.split("\n")[:15])
        bs_kw = sum(1 for kw in ["资产总计", "流动资产合计", "负债合计",
                                  "所有者权益合计"] if kw in text)
        inc_kw = sum(1 for kw in ["营业收入", "营业成本", "净利润",
                                   "利润总额"] if kw in text)
        if bs_kw >= 2 or inc_kw >= 2:
            candidates.append({
                "page_number": page.page_number,
                "head": head.strip(),
                "data_lines": dl,
            })
    return candidates


def _llm_classify_pages(
    report: ParsedReport,
    candidates: list[dict],
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> tuple[list[int], list[int], int]:
    """调用 LLM 对候选页面进行分类，返回 (bs_pages, inc_pages, multiplier)。"""
    if not candidates:
        return [], [], 1

    _consolidated_re = re.compile(
        r"合\s*并\s*(?:及\s*公\s*司\s*)?"
        r"(?:资\s*产\s*负\s*债\s*表|利\s*润\s*(?:及\s*其\s*他\s*综\s*合\s*收\s*益\s*)?表)"
    )
    _any_stmt_re = re.compile(r"资\s*产\s*负\s*债\s*表|利\s*润\s*表")

    def _cand_priority(c):
        head = c["head"]
        if _consolidated_re.search(head):
            return (0, c["page_number"])
        if _any_stmt_re.search(head):
            return (1, c["page_number"])
        return (2, c["page_number"])

    sorted_candidates = sorted(candidates, key=_cand_priority)

    page_summaries = []
    for c in sorted_candidates[:20]:
        pg = c["page_number"]
        text = (report.pages[pg - 1].text or "")[:800]
        page_summaries.append(f"--- 第{pg}页 (数据行{c['data_lines']}行) ---\n{text}")

    prompt_text = _PAGE_CLASSIFY_PROMPT % "\n\n".join(page_summaries)
    client = get_client_for_provider(llm_provider)
    _model = llm_model or get_model_for_provider(llm_provider)

    try:
        raw = chat_completion(
            client, _model,
            [{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        text_clean = raw.strip()
        md = re.search(r"```(?:json)?\s*\n?(.*?)```", text_clean, re.DOTALL)
        if md:
            text_clean = md.group(1).strip()
        jm = re.search(r"\{[\s\S]*\}", text_clean)
        if jm:
            data = json.loads(jm.group())
        else:
            raise ValueError(f"LLM 返回非 JSON: {raw[:200]}")
    except Exception as exc:
        logger.warning("LLM 页面分类失败 (%s)，回退到规则方法", exc)
        return _rule_fallback_classify(report, candidates)

    bs_pages: list[int] = []
    inc_pages: list[int] = []

    for item in data.get("pages", []):
        pg = item.get("page")
        if not isinstance(pg, int):
            continue
        ptype = item.get("type", "")
        is_mother = "母公司" in ptype
        is_other = ptype.strip() in ("其他", "")
        if is_mother or is_other:
            continue
        if "资产负债" in ptype:
            bs_pages.append(pg)
        elif "利润" in ptype:
            inc_pages.append(pg)

    if not bs_pages and not inc_pages:
        logger.warning("LLM 返回空分类结果，回退到规则方法")
        return _rule_fallback_classify(report, candidates)

    unit_text = data.get("consolidated_unit", "元")
    multiplier = {"元": 1, "千元": 1000, "万元": 10000}.get(unit_text, 1)

    def _expand(start_pages: list[int], span: int = 3) -> list[int]:
        out = []
        for pg in start_pages:
            for off in range(span):
                nxt = pg + off
                if nxt > len(report.pages):
                    break
                if off > 0 and _is_in_mother_section(report, nxt):
                    break
                out.append(nxt)
        return sorted(set(out))

    bs_pages = _expand(bs_pages)
    inc_pages = _expand(inc_pages)

    logger.info("LLM 页面分类: BS=%s, IS=%s, 单位=%s(x%d)",
                bs_pages, inc_pages, unit_text, multiplier)
    return bs_pages, inc_pages, multiplier


def _rule_fallback_classify(
    report: ParsedReport,
    candidates: list[dict],
) -> tuple[list[int], list[int], int]:
    """LLM 失败时的规则回退。

    两阶段搜索：先匹配带 "合并" 前缀的报表，再匹配通用标题。
    使用 _is_in_mother_section 回溯检测避免误选母公司报表。
    """
    consolidated_bs_re = re.compile(
        r"合\s*并\s*(?:及\s*公\s*司\s*)?资\s*产\s*负\s*债\s*表"
    )
    consolidated_inc_re = re.compile(
        r"合\s*并\s*(?:及\s*公\s*司\s*)?"
        r"利\s*润\s*(?:及\s*其\s*他\s*综\s*合\s*收\s*益\s*)?表"
    )
    bs_re = re.compile(
        r"(?:合\s*并\s*(?:及\s*公\s*司\s*)?)?资\s*产\s*负\s*债\s*表"
    )
    inc_re = re.compile(
        r"(?:合\s*并\s*(?:及\s*公\s*司\s*)?)?"
        r"利\s*润\s*(?:及\s*其\s*他\s*综\s*合\s*收\s*益\s*)?表"
    )

    bs, inc = [], []
    first_bs_pg = None
    first_inc_pg = None

    # --- 第一阶段：在候选页中优先匹配带 "合并" 前缀的报表 ---
    for c in candidates:
        pg = c["page_number"]
        text = report.pages[pg - 1].text or ""
        head = "\n".join(text.split("\n")[:15])
        if _is_in_mother_section(report, pg):
            continue
        if consolidated_bs_re.search(head):
            if first_bs_pg is None:
                first_bs_pg = pg
                bs.append(pg)
            elif pg <= first_bs_pg + 5:
                bs.append(pg)
        elif consolidated_inc_re.search(head):
            if first_inc_pg is None:
                first_inc_pg = pg
                inc.append(pg)
            elif pg <= first_inc_pg + 5:
                inc.append(pg)

    # --- 第二阶段：无合并标题时，退而搜索通用标题 ---
    if not bs:
        for c in candidates:
            pg = c["page_number"]
            text = report.pages[pg - 1].text or ""
            head = "\n".join(text.split("\n")[:15])
            if _is_in_mother_section(report, pg):
                continue
            if bs_re.search(head):
                if first_bs_pg is None:
                    first_bs_pg = pg
                    bs.append(pg)
                elif pg <= first_bs_pg + 5:
                    bs.append(pg)
    if not inc:
        for c in candidates:
            pg = c["page_number"]
            text = report.pages[pg - 1].text or ""
            head = "\n".join(text.split("\n")[:15])
            if _is_in_mother_section(report, pg):
                continue
            if inc_re.search(head):
                if first_inc_pg is None:
                    first_inc_pg = pg
                    inc.append(pg)
                elif pg <= first_inc_pg + 5:
                    inc.append(pg)

    section_start = _find_financial_section_start(report)

    # --- 候选页都未命中时，扫描全部页面 ---
    if not bs:
        for page in report.pages:
            if page.page_number < section_start:
                continue
            if _is_in_mother_section(report, page.page_number):
                continue
            t = page.text or ""
            head = "\n".join(t.split("\n")[:15])
            if bs_re.search(head) and _count_data_lines(t) >= 3:
                bs.append(page.page_number)
                break
    if not inc:
        for page in report.pages:
            if page.page_number < section_start:
                continue
            if _is_in_mother_section(report, page.page_number):
                continue
            t = page.text or ""
            head = "\n".join(t.split("\n")[:15])
            if inc_re.search(head) and _count_data_lines(t) >= 3:
                inc.append(page.page_number)
                break

    if not inc and bs:
        last_bs = max(bs)
        for off in range(1, 12):
            pg = last_bs + off
            if pg > len(report.pages):
                break
            if _is_in_mother_section(report, pg):
                break
            t = report.pages[pg - 1].text or ""
            head = "\n".join(t.split("\n")[:15])
            kw_count = sum(1 for kw in ["营业收入", "营业成本", "净利润",
                                         "利润总额"] if kw in t)
            if (inc_re.search(head) or kw_count >= 2) and _count_data_lines(t) >= 3:
                inc.append(pg)
                break
    if not bs and inc:
        first_inc = min(inc)
        for off in range(1, 12):
            pg = first_inc - off
            if pg < 1:
                break
            if _is_in_mother_section(report, pg):
                break
            t = report.pages[pg - 1].text or ""
            head = "\n".join(t.split("\n")[:15])
            kw_count = sum(1 for kw in ["资产总计", "流动资产合计", "负债合计",
                                         "所有者权益合计"] if kw in t)
            if (bs_re.search(head) or kw_count >= 2) and _count_data_lines(t) >= 3:
                bs.append(pg)
                break

    def _expand(pages):
        out = []
        for pg in pages:
            for off in range(3):
                nxt = pg + off
                if nxt > len(report.pages):
                    break
                if off > 0 and _is_in_mother_section(report, nxt):
                    break
                out.append(nxt)
        return sorted(set(out))

    bs = _expand(bs)
    inc = _expand(inc)

    multiplier = 1
    unit_re = re.compile(r"单位[:：]\s*(千元|万元|元)")
    for pg in sorted(set(bs + inc)):
        for off in range(-3, 2):
            pg2 = pg + off
            if pg2 < max(1, section_start - 1) or pg2 > len(report.pages):
                continue
            m = unit_re.search(report.pages[pg2 - 1].text or "")
            if m:
                multiplier = {"元": 1, "千元": 1000, "万元": 10000}[m.group(1)]
                break
        if multiplier != 1:
            break

    if multiplier == 1 and inc:
        rev_val = None
        for pg in inc:
            text = report.pages[pg - 1].text or ""
            for ln in text.split("\n"):
                if "营业收入" in ln:
                    nums = _parse_numbers_from_line(ln)
                    if nums:
                        rev_val = nums[0]
                        break
            if rev_val:
                break
        if rev_val and rev_val > 0:
            for page in report.pages:
                m = re.search(
                    r"营业收入[（(]元[）)]\s*([\d,]+\.?\d*)", page.text or "")
                if m:
                    summary = float(m.group(1).replace(",", ""))
                    ratio = summary / rev_val
                    if 800 < ratio < 1200:
                        multiplier = 1000
                    elif 8000 < ratio < 12000:
                        multiplier = 10000
                    break

    logger.info("规则回退分类: BS=%s, IS=%s, 单位x%d", bs, inc, multiplier)
    return bs, inc, multiplier


# ---------------------------------------------------------------------------
# 第二步：正则精确提取数值
# ---------------------------------------------------------------------------

def _parse_numbers_from_line(
    line: str, *, financial_only: bool = False,
) -> list[float]:
    """从行文本中提取数值。

    financial_only=True 时只返回高可信度财务数值（带千位逗号或绝对值>=10000），
    排除注释编号等噪声。
    """
    tokens: list[tuple[str, str, str]] = _NUMBER_TOKEN_RE.findall(line)
    values: list[float] = []
    for paren_neg, dash_neg, pos_part in tokens:
        raw = paren_neg or dash_neg or pos_part
        raw_clean = raw.replace(",", "").strip()
        if not raw_clean:
            continue
        try:
            val = float(raw_clean)
        except ValueError:
            continue
        if paren_neg or dash_neg:
            val = -val

        has_comma = bool(re.search(r"\d,\d{3}", raw))

        if abs(val) < 100 and not has_comma:
            continue
        if abs(val) < 10000 and not has_comma and "." not in raw:
            continue

        if financial_only and not has_comma and abs(val) < 10000:
            continue

        values.append(val)
    return values


def _select_value(numbers: list[float], period: str) -> float | None:
    """从数值列表中按 period 选取对应值。"""
    if not numbers:
        return None
    idx = 1 if period == "prior" else 0
    return numbers[idx] if idx < len(numbers) else numbers[0]


def _search_value_in_pages(
    report: ParsedReport,
    page_numbers: list[int],
    aliases: list[str],
    period: str = "current",
    *,
    _allow_expand: bool = True,
) -> float | None:
    for pg_num in page_numbers:
        if pg_num < 1 or pg_num > len(report.pages):
            continue
        page = report.pages[pg_num - 1]
        if not page.text:
            continue
        for line in page.text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            item_name = _extract_item_name(line_stripped)
            if not item_name:
                continue
            for alias in aliases:
                alias_core = _PREFIX_RE.sub("", alias.replace(" ", "")).strip()
                if item_name.replace(" ", "") != alias_core:
                    continue
                numbers = _parse_numbers_from_line(
                    line_stripped, financial_only=True)
                if not numbers:
                    numbers = _parse_numbers_from_line(line_stripped)
                val = _select_value(numbers, period)
                if val is not None:
                    return val

    for pg_num in page_numbers:
        if pg_num < 1 or pg_num > len(report.pages):
            continue
        page = report.pages[pg_num - 1]
        if not page.text:
            continue
        lines = [ln.strip() for ln in page.text.split("\n") if ln.strip()]
        for i in range(len(lines) - 1):
            merged3 = lines[i] + lines[i + 1]
            if i + 2 < len(lines):
                merged3 += lines[i + 2]
            for alias in aliases:
                alias_core = _PREFIX_RE.sub("", alias.replace(" ", "")).strip()
                merged_clean = merged3.replace(" ", "").replace("\n", "")
                pos = merged_clean.find(alias_core)
                if pos < 0 or pos > 30:
                    continue
                if pos > 0 and "\u4e00" <= merged_clean[pos - 1] <= "\u9fff":
                    continue
                after = pos + len(alias_core)
                if after < len(merged_clean):
                    ch = merged_clean[after]
                    if "\u4e00" <= ch <= "\u9fff" and ch not in "合计":
                        continue
                start_line = i
                for off in range(min(3, len(lines) - i)):
                    if alias_core in lines[i + off].replace(" ", ""):
                        start_line = i + off
                        break
                span = lines[start_line:min(i + 3, len(lines))]
                all_nums: list[float] = []
                for _si, ln in enumerate(span):
                    if _si > 0:
                        ln_item = _extract_item_name(ln.strip())
                        if (ln_item
                                and ln_item.replace(" ", "") != alias_core
                                and any("\u4e00" <= c <= "\u9fff"
                                        for c in ln_item)
                                and len(ln_item) >= 2
                                and ln_item not in ("合计", "总计", "小计")):
                            continue
                    all_nums.extend(
                        _parse_numbers_from_line(ln, financial_only=True))
                if not all_nums:
                    for _si, ln in enumerate(span):
                        if _si > 0:
                            ln_item = _extract_item_name(ln.strip())
                            if (ln_item
                                    and ln_item.replace(" ", "") != alias_core
                                    and any("\u4e00" <= c <= "\u9fff"
                                            for c in ln_item)
                                    and len(ln_item) >= 2
                                    and ln_item not in (
                                        "合计", "总计", "小计")):
                                continue
                        all_nums.extend(_parse_numbers_from_line(ln))
                val = _select_value(all_nums, period)
                if val is not None:
                    return val

    if _allow_expand and page_numbers:
        nearby = sorted(set(
            pg + off
            for pg in page_numbers
            for off in [-2, -1, 1, 2]
            if (pg + off) not in page_numbers
            and 1 <= pg + off <= len(report.pages)
            and not _is_in_mother_section(report, pg + off)
        ))
        if nearby:
            return _search_value_in_pages(
                report, nearby, aliases, period, _allow_expand=False)
    return None


def _search_equity_fallback(
    report: ParsedReport,
    page_numbers: list[int],
    period: str = "current",
) -> float | None:
    for pg_num in page_numbers:
        if pg_num < 1 or pg_num > len(report.pages):
            continue
        page = report.pages[pg_num - 1]
        if not page.text:
            continue
        lines = [ln.strip() for ln in page.text.split("\n") if ln.strip()]
        found_minority = False
        for i, line in enumerate(lines):
            if "\u5c11\u6570\u80a1\u4e1c\u6743\u76ca" in line:
                found_minority = True
                continue
            if not found_minority:
                continue
            nums = _parse_numbers_from_line(line)
            has_text = any("\u4e00" <= ch <= "\u9fff" for ch in line)
            if nums and not has_text:
                idx = 1 if period == "prior" else 0
                return nums[idx] if idx < len(nums) else nums[0]
            if "\u5408\u8ba1" in line and nums:
                idx = 1 if period == "prior" else 0
                return nums[idx] if idx < len(nums) else nums[0]
            if has_text and "\u6743\u76ca" not in line and "\u5408\u8ba1" not in line:
                found_minority = False
    return None


# ---------------------------------------------------------------------------
# LLM 辅助提取与异常检测（正则提取的兜底校正层）
# ---------------------------------------------------------------------------

_LLM_EXTRACT_PROMPT = """\
你是上市公司年报财务数据提取专家。请从以下合并报表原文中精确提取财务数据。

关键要求：
1. 只提取【合并报表】数据，忽略母公司报表
2. "净利润"取合并利润表的净利润总额行（通常标注为"五、净利润"），\
不是"其中：归属于母公司所有者的净利润"子项
3. 资产负债表第一列是"期末余额"（本期），第二列是"期初余额"
4. 利润表第一列是"本期金额"，第二列是"上期金额"
5. 请分别提取两列数据：本期/期末 和 上期/期初
6. 注意报表的单位（在表头如"单位：元"或"单位：万元"），将所有金额换算为【元】

合并资产负债表：
%s

合并利润表：
%s

返回纯JSON（无markdown标记、无注释），金额统一为元：
{"营业收入":数值,"营业成本":数值,"净利润":数值,"利润总额":数值,\
"财务费用":数值,"所得税费用":数值,"资产总计":数值,"流动资产合计":数值,\
"流动负债合计":数值,"负债合计":数值,"所有者权益合计":数值,\
"存货":数值,"应收账款":数值,"未分配利润":数值,"盈余公积":数值,\
"期初资产总计":数值,"期初所有者权益合计":数值,\
"期初应收账款":数值,"期初存货":数值}
缺失项填null。"""


def _detect_anomalies(values: dict[str, float | None]) -> list[tuple[str, str]]:
    """检测提取值中的财务逻辑异常。返回 [(字段名, 原因)] 列表。"""
    anomalies: list[tuple[str, str]] = []
    rev = values.get("营业收入")
    cost = values.get("营业成本")
    net = values.get("净利润")
    profit_bt = values.get("利润总额")
    ta = values.get("资产总计")
    tl = values.get("负债合计")
    eq = values.get("所有者权益合计")

    if rev and net and abs(net) > abs(rev) * 1.0:
        anomalies.append(("净利润", "净利润超过营业收入"))
        anomalies.append(("营业收入", "与净利润矛盾，需验证"))

    if rev and cost and rev > 0 and cost > rev * 3:
        anomalies.append(("营业成本",
                          f"营业成本是营业收入的{cost / rev:.1f}倍"))

    if profit_bt and net and profit_bt > 0 and net > profit_bt * 1.5:
        anomalies.append(("净利润", "净利润远大于利润总额"))

    if ta and tl and eq:
        diff_pct = abs(ta - tl - eq) / abs(ta) if ta != 0 else 0
        if diff_pct > 0.10:
            anomalies.append(("所有者权益合计",
                              f"会计恒等式偏差{diff_pct * 100:.1f}%"))

    critical = ["营业收入", "净利润", "资产总计", "负债合计", "所有者权益合计"]
    missing = [k for k in critical if values.get(k) is None]
    if len(missing) >= 2:
        for m in missing:
            anomalies.append((m, "关键字段缺失"))

    return anomalies


def _llm_extract_financial_values(
    report: ParsedReport,
    bs_pages: list[int],
    inc_pages: list[int],
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> dict[str, float | None]:
    """LLM 直接从报表原文提取财务数值。作为正则提取的兜底校正。"""
    bs_parts: list[str] = []
    for pg in bs_pages:
        if 1 <= pg <= len(report.pages):
            t = report.pages[pg - 1].text or ""
            if t.strip():
                bs_parts.append(f"--- 第{pg}页 ---\n{t}")
    inc_parts: list[str] = []
    for pg in inc_pages:
        if 1 <= pg <= len(report.pages):
            t = report.pages[pg - 1].text or ""
            if t.strip():
                inc_parts.append(f"--- 第{pg}页 ---\n{t}")

    bs_text = "\n".join(bs_parts)[:6000]
    inc_text = "\n".join(inc_parts)[:6000]
    if not bs_text and not inc_text:
        logger.warning("LLM 提取: 无可用页面文本")
        return {}

    prompt = _LLM_EXTRACT_PROMPT % (bs_text, inc_text)
    client = get_client_for_provider(llm_provider)
    _model = llm_model or get_model_for_provider(llm_provider)

    try:
        raw = chat_completion(
            client, _model,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text_clean = raw.strip()
        md = re.search(r"```(?:json)?\s*\n?(.*?)```", text_clean, re.DOTALL)
        if md:
            text_clean = md.group(1).strip()
        jm = re.search(r"\{[\s\S]*\}", text_clean)
        if not jm:
            raise ValueError(f"LLM 返回非JSON: {raw[:200]}")
        data = json.loads(jm.group())
    except Exception as exc:
        logger.warning("LLM 财务数值提取失败: %s", exc)
        return {}

    result: dict[str, float | None] = {}
    for key in ITEM_ALIASES:
        val = data.get(key)
        if val is not None:
            try:
                result[key] = float(val)
            except (ValueError, TypeError):
                pass

    ok = sum(1 for v in result.values() if v is not None)
    logger.info("LLM 提取: %d/%d 个字段成功", ok, len(ITEM_ALIASES))
    return result


# ---------------------------------------------------------------------------
# 第三步：综合提取 + 会计恒等式校验
# ---------------------------------------------------------------------------

_MOTHER_RE = re.compile(r"母\s*公\s*司")

_MOTHER_SECTION_TITLE_RE = re.compile(
    r"母\s*公\s*司\s*(?:资\s*产\s*负\s*债|利\s*润|现\s*金\s*流\s*量|所\s*有\s*者)"
)

_MOTHER_ATTRIBUTION_RE = re.compile(
    r"归属[于至]?\s*母\s*公\s*司\s*[所股]"
)


def _has_mother_indicator(text: str) -> bool:
    """检测文本是否包含母公司区域标记（排除'归属于母公司所有者'等误触发）。"""
    cleaned = _MOTHER_ATTRIBUTION_RE.sub("", text)
    return bool(_MOTHER_RE.search(cleaned))


_CONSOLIDATED_TITLE_RE = re.compile(
    r"合\s*并\s*(?:及\s*公\s*司\s*)?"
    r"(?:资\s*产\s*负\s*债|利\s*润|现\s*金\s*流\s*量)"
)


def _is_in_mother_section(report: ParsedReport, pg_num: int) -> bool:
    """判断页面是否属于母公司报表区域。

    优先级：
    1. 本页有母公司报表标题 → True
    2. 本页有合并报表标题 → False（即使前面有母公司区域）
    3. 前15行有母公司标记（排除归属措辞） → True
    4. 向前回溯 7 页判断上下文
    """
    text = report.pages[pg_num - 1].text or ""
    text_cleaned = _MOTHER_ATTRIBUTION_RE.sub("", text)
    if _MOTHER_SECTION_TITLE_RE.search(text_cleaned):
        return True
    if _CONSOLIDATED_TITLE_RE.search(text):
        return False
    head = "\n".join(text.split("\n")[:15])
    if _has_mother_indicator(head[:500]):
        return True
    for off in range(1, 8):
        prev_pg = pg_num - off
        if prev_pg < 1:
            break
        prev_text = report.pages[prev_pg - 1].text or ""
        if _CONSOLIDATED_TITLE_RE.search(prev_text):
            return False
        prev_cleaned = _MOTHER_ATTRIBUTION_RE.sub("", prev_text)
        if _MOTHER_SECTION_TITLE_RE.search(prev_cleaned):
            return True
        prev_head = "\n".join(prev_text.split("\n")[:15])
        if _ANY_FINANCIAL_RE.search(prev_head) and not _has_mother_indicator(prev_head[:500]):
            return False
        if _has_mother_indicator(prev_head[:500]):
            return True
    return False


def _discover_nearby_pages(
    report: ParsedReport,
    known_pages: list[int],
    target_type: str,
) -> list[int]:
    """当已知BS(或IS)页但IS(或BS)页缺失时，从已知页附近搜索。"""
    if target_type == "IS":
        kw_list = ["营业收入", "营业成本", "净利润", "利润总额"]
        search_range = range(1, 12)
        base = max(known_pages)
    else:
        kw_list = ["资产总计", "流动资产合计", "负债合计", "所有者权益合计"]
        search_range = range(1, 12)
        base = min(known_pages)

    found: list[int] = []
    for off in search_range:
        pg = base + off if target_type == "IS" else base - off
        if pg < 1 or pg > len(report.pages):
            break
        t = report.pages[pg - 1].text or ""
        if _is_in_mother_section(report, pg):
            continue
        kw_count = sum(1 for kw in kw_list if kw in t)
        if kw_count >= 2 and _count_data_lines(t) >= 3:
            found.append(pg)
            break
        if re.search(r"合\s*并\s*(?:及\s*公\s*司\s*)?利\s*润", t) and target_type == "IS":
            if _count_data_lines(t) >= 2:
                found.append(pg)
                break
    return found


def _extract_key_values(
    report: ParsedReport,
    llm_provider: str = "tengri",
    llm_model: str = "",
) -> dict[str, float | None]:
    """LLM辅助页面识别 + 正则精确提取 + 单位标准化 + 会计恒等式校验。"""

    candidates = _get_candidate_pages(report)
    bs_pages, inc_pages, multiplier = _llm_classify_pages(
        report, candidates, llm_provider=llm_provider, llm_model=llm_model,
    )

    if not inc_pages and bs_pages:
        extra = _discover_nearby_pages(report, bs_pages, "IS")
        if extra:
            inc_pages = list(extra)
            for pg in extra:
                for off in range(1, 3):
                    nxt = pg + off
                    if nxt <= len(report.pages) and nxt not in inc_pages:
                        if _is_in_mother_section(report, nxt):
                            break
                        inc_pages.append(nxt)
            inc_pages = sorted(set(inc_pages))
            logger.info("从BS页附近发现IS页: %s", inc_pages)
    if not bs_pages and inc_pages:
        extra = _discover_nearby_pages(report, inc_pages, "BS")
        if extra:
            bs_pages = sorted(set(bs_pages + extra))
            for pg in extra:
                for off in range(1, 3):
                    nxt = pg + off
                    if nxt <= len(report.pages) and nxt not in bs_pages:
                        bs_pages.append(nxt)
            bs_pages = sorted(set(bs_pages))
            logger.info("从IS页附近发现BS页: %s", bs_pages)

    # --- 验证 IS 页是否包含营业收入，否则向前搜索补充 ---
    if inc_pages:
        has_revenue = any(
            "营业收入" in (report.pages[pg - 1].text or "")
            or "营业总收入" in (report.pages[pg - 1].text or "")
            for pg in inc_pages
        )
        if not has_revenue:
            first_is = min(inc_pages)
            bs_max = max(bs_pages) if bs_pages else 0
            logger.warning("IS页(%s)中未找到营业收入，向前搜索补充", inc_pages)
            for pg in range(first_is - 1, bs_max, -1):
                if _is_in_mother_section(report, pg):
                    break
                t = report.pages[pg - 1].text or ""
                has_title = bool(re.search(
                    r"合\s*并\s*(?:及\s*公\s*司\s*)?利\s*润", t))
                has_rev = "营业收入" in t or "营业总收入" in t
                if (has_title or has_rev) and _count_data_lines(t) >= 2:
                    inc_pages.append(pg)
                    for mid_pg in range(pg + 1, first_is):
                        if mid_pg not in inc_pages:
                            inc_pages.append(mid_pg)
                    inc_pages = sorted(set(inc_pages))
                    logger.info("  补充IS页: %s", inc_pages)
                    break

    logger.info("合并报表页码 - BS: %s, IS: %s, 单位x%d", bs_pages, inc_pages, multiplier)

    values: dict[str, float | None] = {}
    extraction_log: list[str] = []

    income_items = {"营业收入", "营业成本", "净利润", "利润总额",
                    "财务费用", "所得税费用"}
    all_stmt_pages = sorted(set(bs_pages + inc_pages))

    for item_key, aliases in ITEM_ALIASES.items():
        is_prior = item_key.startswith("期初")
        period = "prior" if is_prior else "current"
        actual_key = item_key.replace("期初", "") if is_prior else item_key
        if actual_key in income_items:
            search_pages = inc_pages if inc_pages else all_stmt_pages
        else:
            search_pages = bs_pages if bs_pages else all_stmt_pages
        val = _search_value_in_pages(report, search_pages, aliases, period)
        values[item_key] = val
        if val is not None:
            extraction_log.append(f"  [OK] {item_key} = {val:,.0f}")
        else:
            extraction_log.append(f"  [--] {item_key}: 未找到")

    if values.get("财务费用") is None and inc_pages:
        fin_income_aliases = ["财务收入", "财务费用净额"]
        val = _search_value_in_pages(report, inc_pages, fin_income_aliases, "current")
        if val is not None:
            values["财务费用"] = -val
            extraction_log.append(f"  [FLIP] 财务费用 = {-val:,.0f} (由财务收入取反)")

    bs_search = bs_pages or all_stmt_pages[:6]
    for eq_key, eq_period in [("所有者权益合计", "current"), ("期初所有者权益合计", "prior")]:
        if values.get(eq_key) is not None:
            continue
        val = _search_equity_fallback(report, bs_search, eq_period)
        if val is not None:
            values[eq_key] = val
            extraction_log.append(f"  [FB] {eq_key} = {val:,.0f} (跨行回退)")

    for eq_key, eq_period in [("所有者权益合计", "current"), ("期初所有者权益合计", "prior")]:
        if values.get(eq_key) is not None:
            continue
        if eq_period == "current":
            ta_val = values.get("资产总计")
            liab_val = values.get("负债合计")
        else:
            ta_val = values.get("期初资产总计")
            liab_val = _search_value_in_pages(
                report, bs_search, ITEM_ALIASES["负债合计"], "prior")
        if ta_val is not None and liab_val is not None and ta_val > liab_val > 0:
            computed = ta_val - liab_val
            values[eq_key] = computed
            extraction_log.append(f"  [CALC] {eq_key} = {computed:,.0f} (资产-负债)")

    if multiplier != 1:
        for key in values:
            if values[key] is not None:
                values[key] = values[key] * multiplier
        extraction_log.append(f"  [单位] x{multiplier} 标准化为元")

    for cost_key in ("营业成本", "所得税费用"):
        if values.get(cost_key) is not None and values[cost_key] < 0:
            values[cost_key] = abs(values[cost_key])
            extraction_log.append(f"  [FIX] {cost_key}: 负值已修正为绝对值")

    _pt_chk = values.get("利润总额")
    _tax_chk = values.get("所得税费用")
    _ni_chk = values.get("净利润")
    if (_pt_chk is not None and _tax_chk is not None and _ni_chk is not None
            and abs(_ni_chk) > 0 and _tax_chk > 0):
        _err_pos = abs((_pt_chk - _tax_chk) - _ni_chk)
        _err_neg = abs((_pt_chk + _tax_chk) - _ni_chk)
        if _err_neg < _err_pos and _err_neg / abs(_ni_chk) < 0.01:
            values["所得税费用"] = -_tax_chk
            extraction_log.append(
                f"  [FIX] 所得税费用: 会计恒等式校正 {_tax_chk:,.0f} -> {-_tax_chk:,.0f}")

    _MIN_MAJOR = 1_000_000
    for key in ("营业收入", "营业成本", "净利润", "利润总额", "资产总计",
                "负债合计", "所有者权益合计"):
        val = values.get(key)
        if val is not None and abs(val) < _MIN_MAJOR:
            extraction_log.append(
                f"  [FIX] {key} = {val:,.0f} 异常偏小(<100万)，已清除")
            values[key] = None

    _MIN_MINOR = 10_000
    for key in values:
        if key in ("营业收入", "营业成本", "净利润", "利润总额", "资产总计",
                   "负债合计", "所有者权益合计"):
            continue
        val = values.get(key)
        if val is not None and abs(val) < _MIN_MINOR:
            extraction_log.append(
                f"  [FIX] {key} = {val:,.0f} 异常偏小(<1万)，已清除")
            values[key] = None

    rev = values.get("营业收入")
    cost = values.get("营业成本")
    if rev and cost and min(abs(rev), abs(cost)) > 0:
        ratio = max(abs(rev), abs(cost)) / min(abs(rev), abs(cost))
        if ratio > 50:
            smaller_key = "营业成本" if abs(cost) < abs(rev) else "营业收入"
            extraction_log.append(
                f"  [FIX] 营收与成本差{ratio:.0f}倍，清除异常的{smaller_key}")
            values[smaller_key] = None

    for curr_key, prior_key in [
        ("资产总计", "期初资产总计"),
        ("所有者权益合计", "期初所有者权益合计"),
    ]:
        curr, prior = values.get(curr_key), values.get(prior_key)
        if curr and prior and min(abs(curr), abs(prior)) > 0:
            ratio = max(abs(curr), abs(prior)) / min(abs(curr), abs(prior))
            if ratio > 20:
                extraction_log.append(
                    f"  [FIX] {curr_key}与{prior_key}相差{ratio:.0f}倍，清除期初值")
                values[prior_key] = None

    for eq_key, eq_period in [("所有者权益合计", "current"), ("期初所有者权益合计", "prior")]:
        if values.get(eq_key) is not None:
            continue
        if eq_period == "current":
            ta_val = values.get("资产总计")
            liab_val = values.get("负债合计")
        else:
            ta_val = values.get("期初资产总计")
            liab_val = _search_value_in_pages(
                report, bs_search, ITEM_ALIASES["负债合计"], "prior")
        if ta_val is not None and liab_val is not None and ta_val > liab_val > 0:
            computed = ta_val - liab_val
            values[eq_key] = computed
            extraction_log.append(f"  [CALC] {eq_key} = {computed:,.0f} (资产-负债，清理后回补)")

    if extraction_log:
        logger.info("财务数值提取结果:\n%s", "\n".join(extraction_log))

    ta = values.get("资产总计")
    liab = values.get("负债合计")
    eq = values.get("所有者权益合计")
    if ta and liab and eq:
        computed_eq = ta - liab
        diff_pct = abs(ta - liab - eq) / abs(ta) if ta != 0 else 0
        if diff_pct <= 0.01:
            logger.info("会计恒等式校验通过 (偏差%.4f%%)", diff_pct * 100)
        elif diff_pct > 0.05 and computed_eq > 0:
            logger.warning(
                "会计恒等式偏差%.2f%%, 修正权益: %s -> %s",
                diff_pct * 100, f"{eq:,.0f}", f"{computed_eq:,.0f}")
            values["所有者权益合计"] = computed_eq
        else:
            logger.warning(
                "会计恒等式校验失败: 资产(%s) != 负债(%s)+权益(%s), 偏差%.2f%%",
                f"{ta:,.0f}", f"{liab:,.0f}", f"{eq:,.0f}", diff_pct * 100)

    # --- LLM 辅助校正：检测异常后调用 LLM 重新提取 ---
    anomalies = _detect_anomalies(values)
    pre_llm_count = sum(1 for v in values.values() if v is not None)
    need_llm = bool(anomalies) or pre_llm_count < 10

    if need_llm:
        if anomalies:
            logger.warning("检测到 %d 个提取异常，启用 LLM 辅助校正:", len(anomalies))
            for _fld, _reason in anomalies:
                logger.warning("  - %s: %s", _fld, _reason)
        else:
            logger.warning("提取率偏低 (%d/19)，启用 LLM 辅助补充", pre_llm_count)

        llm_vals = _llm_extract_financial_values(
            report, bs_pages, inc_pages,
            llm_provider=llm_provider, llm_model=llm_model,
        )
        if llm_vals:
            anomalous_keys = {a[0] for a in anomalies}
            for key, llm_val in llm_vals.items():
                if llm_val is None:
                    continue
                if abs(llm_val) < 1 and key != "财务费用":
                    continue
                if key in anomalous_keys:
                    old = values.get(key)
                    values[key] = llm_val
                    logger.info("  [LLM-修正] %s: %s -> %s", key,
                                f"{old:,.0f}" if old else "None",
                                f"{llm_val:,.0f}")
                elif values.get(key) is None:
                    values[key] = llm_val
                    logger.info("  [LLM-补充] %s = %s", key, f"{llm_val:,.0f}")

            ta_v = values.get("资产总计")
            li_v = values.get("负债合计")
            eq_v = values.get("所有者权益合计")
            if ta_v and li_v and eq_v:
                dp = abs(ta_v - li_v - eq_v) / abs(ta_v) if ta_v != 0 else 0
                if dp > 0.05:
                    comp = ta_v - li_v
                    if comp > 0:
                        values["所有者权益合计"] = comp
                        logger.info("  [LLM] 会计恒等式校正: 权益=%s",
                                    f"{comp:,.0f}")

    # --- 全页面搜索兜底：关键字段缺失或异常时暴力搜索全文 ---
    _section_start = _find_financial_section_start(report)
    _rev_check = values.get("营业收入")
    _ta_check = values.get("资产总计")
    _rev_suspicious = (_rev_check is None) or (
        _rev_check is not None and _ta_check is not None
        and _ta_check > 1e10 and _rev_check < _ta_check * 0.05
    )
    if _rev_suspicious:
        if _rev_check is not None:
            logger.warning("营业收入(%s)相对资产总计(%s)异常偏小(%.1f%%)，触发全页搜索",
                           f"{_rev_check:,.0f}", f"{_ta_check:,.0f}",
                           _rev_check / _ta_check * 100 if _ta_check else 0)
            for _is_key in ["营业收入", "营业成本", "净利润", "利润总额",
                            "财务费用", "所得税费用"]:
                values[_is_key] = None
        for pg_num in range(_section_start, len(report.pages) + 1):
            if _is_in_mother_section(report, pg_num):
                continue
            t = report.pages[pg_num - 1].text or ""
            if ("营业收入" in t or "营业总收入" in t) and _count_data_lines(t) >= 3:
                val = _search_value_in_pages(
                    report, [pg_num], ITEM_ALIASES["营业收入"], "current")
                if val is not None and abs(val) > 1e8:
                    values["营业收入"] = val
                    logger.info("  [全页搜索] 营业收入 = %s (第%d页)",
                                f"{val:,.0f}", pg_num)
                    for item in ["营业成本", "净利润", "利润总额",
                                 "财务费用", "所得税费用"]:
                        if values.get(item) is not None:
                            continue
                        nearby = [pg_num + off for off in range(4)
                                  if 1 <= pg_num + off <= len(report.pages)]
                        v2 = _search_value_in_pages(
                            report, nearby, ITEM_ALIASES[item], "current")
                        if v2 is not None and abs(v2) >= _MIN_MAJOR:
                            values[item] = v2
                            logger.info("  [全页搜索] %s = %s", item,
                                        f"{v2:,.0f}")
                    break

    _rev_after = values.get("营业收入")
    _ta_after = values.get("资产总计")
    _bs_data_suspicious = (
        _rev_after is not None and _ta_after is not None
        and _rev_after > _ta_after * 2 and _ta_after > 1e9
    )
    if _bs_data_suspicious:
        logger.warning("营业收入(%s)远大于资产总计(%s)，BS数据可能为母公司，触发BS全页搜索",
                       f"{_rev_after:,.0f}", f"{_ta_after:,.0f}")
        for k in ["资产总计", "负债合计", "所有者权益合计", "流动资产合计",
                   "流动负债合计", "存货", "应收账款"]:
            values[k] = None

    _bs_missing_items = [item for item in
                         ["资产总计", "负债合计", "所有者权益合计",
                          "存货", "应收账款"]
                         if values.get(item) is None]
    if _bs_missing_items:
        _found_bs_pg = None
        for pg_num in range(_section_start, len(report.pages) + 1):
            if _is_in_mother_section(report, pg_num):
                continue
            t = report.pages[pg_num - 1].text or ""
            if "资产总计" in t and _count_data_lines(t) >= 3:
                val = _search_value_in_pages(
                    report, [pg_num], ITEM_ALIASES["资产总计"], "current")
                if val is not None and abs(val) > 1e9:
                    if values.get("资产总计") is None:
                        values["资产总计"] = val
                        logger.info("  [全页搜索] 资产总计 = %s (第%d页)",
                                    f"{val:,.0f}", pg_num)
                    _found_bs_pg = pg_num
                    break
        if _found_bs_pg:
            nearby = sorted(set(
                _found_bs_pg + off for off in range(-3, 8)
                if 1 <= _found_bs_pg + off <= len(report.pages)
                and not _is_in_mother_section(report, _found_bs_pg + off)
            ))
            for item in ["负债合计", "所有者权益合计", "流动资产合计",
                         "流动负债合计", "存货", "应收账款",
                         "未分配利润", "盈余公积"]:
                if values.get(item) is not None:
                    continue
                v2 = _search_value_in_pages(
                    report, nearby, ITEM_ALIASES[item], "current")
                if v2 is not None and abs(v2) >= _MIN_MINOR:
                    values[item] = v2
                    logger.info("  [全页搜索] %s = %s", item, f"{v2:,.0f}")

    # --- 专项搜索：未分配利润/盈余公积缺失时，逐行扫描权益区域 ---
    _retained_missing = [k for k in ("未分配利润", "盈余公积")
                         if values.get(k) is None]
    if _retained_missing:
        _mother_title_re = re.compile(
            r"母\s*公\s*司\s*(?:资\s*产\s*负\s*债|利\s*润|现\s*金\s*流\s*量)")
        _eq_scan_range = sorted(set(
            pg + off
            for pg in (bs_pages or all_stmt_pages[:6])
            for off in range(-1, 12)
            if 1 <= pg + off <= len(report.pages)
        ))
        for item in _retained_missing:
            aliases = ITEM_ALIASES[item]
            found_val = None
            for pg_num in _eq_scan_range:
                page = report.pages[pg_num - 1]
                if not page.text:
                    continue
                for line in page.text.split("\n"):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if _mother_title_re.search(line_stripped):
                        break
                    item_name = _extract_item_name(line_stripped)
                    if not item_name:
                        continue
                    for alias in aliases:
                        alias_core = _PREFIX_RE.sub(
                            "", alias.replace(" ", "")).strip()
                        if item_name.replace(" ", "") != alias_core:
                            continue
                        numbers = _parse_numbers_from_line(
                            line_stripped, financial_only=True)
                        if not numbers:
                            numbers = _parse_numbers_from_line(line_stripped)
                        val = _select_value(numbers, "current")
                        if val is not None:
                            found_val = val
                            break
                    if found_val is not None:
                        break
                if found_val is not None:
                    break
            if found_val is not None:
                val_scaled = (found_val * multiplier
                              if multiplier != 1 else found_val)
                if abs(val_scaled) >= _MIN_MINOR:
                    values[item] = val_scaled
                    logger.info("  [权益专项] %s = %s (第%d页)",
                                item, f"{val_scaled:,.0f}", pg_num)

    # --- 专项搜索：负债合计/所有者权益合计仍缺失时，从BS页后方页面搜索 ---
    _liab_eq_still_missing = (
        values.get("负债合计") is None or values.get("所有者权益合计") is None
    )
    if _liab_eq_still_missing and values.get("资产总计") is not None:
        _anchor_pg = None
        for pg_num in (bs_pages or all_stmt_pages):
            if pg_num < 1 or pg_num > len(report.pages):
                continue
            t = report.pages[pg_num - 1].text or ""
            if "资产总计" in t:
                _anchor_pg = pg_num
                break
        if _anchor_pg is None:
            for pg_num in range(_section_start, len(report.pages) + 1):
                t = report.pages[pg_num - 1].text or ""
                if "资产总计" in t and _count_data_lines(t) >= 2:
                    _anchor_pg = pg_num
                    break
        if _anchor_pg:
            _ext_pages = [
                _anchor_pg + off for off in range(1, 10)
                if 1 <= _anchor_pg + off <= len(report.pages)
                and not _is_in_mother_section(report, _anchor_pg + off)
            ]
            logger.info("负债/权益缺失，从资产总计(第%d页)向后扩展搜索: %s",
                        _anchor_pg, _ext_pages)
            for item in ["负债合计", "所有者权益合计",
                         "流动负债合计", "未分配利润", "盈余公积"]:
                if values.get(item) is not None:
                    continue
                v2 = _search_value_in_pages(
                    report, _ext_pages, ITEM_ALIASES[item], "current")
                if v2 is not None:
                    v2_scaled = v2 * multiplier if multiplier != 1 else v2
                    threshold = _MIN_MAJOR if item in ("负债合计", "所有者权益合计") else _MIN_MINOR
                    if abs(v2_scaled) >= threshold:
                        values[item] = v2_scaled
                        logger.info("  [扩展搜索] %s = %s (第%s页范围)",
                                    item, f"{v2_scaled:,.0f}", _ext_pages)
            for eq_key in ["期初所有者权益合计"]:
                if values.get(eq_key) is not None:
                    continue
                v2 = _search_value_in_pages(
                    report, _ext_pages, ITEM_ALIASES[eq_key], "prior")
                if v2 is not None:
                    v2_scaled = v2 * multiplier if multiplier != 1 else v2
                    if abs(v2_scaled) >= _MIN_MAJOR:
                        values[eq_key] = v2_scaled
                        logger.info("  [扩展搜索] %s = %s", eq_key, f"{v2_scaled:,.0f}")
            if values.get("所有者权益合计") is None:
                ta_v = values.get("资产总计")
                tl_v = values.get("负债合计")
                if ta_v is not None and tl_v is not None and ta_v > tl_v > 0:
                    values["所有者权益合计"] = ta_v - tl_v
                    logger.info("  [扩展推导] 所有者权益合计 = %s (资产-负债)",
                                f"{values['所有者权益合计']:,.0f}")
            if values.get("负债合计") is None:
                ta_v = values.get("资产总计")
                eq_v = values.get("所有者权益合计")
                if ta_v is not None and eq_v is not None and ta_v > eq_v > 0:
                    values["负债合计"] = ta_v - eq_v
                    logger.info("  [扩展推导] 负债合计 = %s (资产-权益)",
                                f"{values['负债合计']:,.0f}")

    success_count = sum(1 for v in values.values() if v is not None)
    if success_count < 5:
        logger.warning("提取率极低(%d/19), 使用所有候选页面重试", success_count)
        all_cand_pages = sorted(set(c["page_number"] for c in candidates))
        for item_key, aliases in ITEM_ALIASES.items():
            if values.get(item_key) is not None:
                continue
            is_prior = item_key.startswith("期初")
            period = "prior" if is_prior else "current"
            val = _search_value_in_pages(
                report, all_cand_pages, aliases, period, _allow_expand=False)
            if val is not None:
                val = val * multiplier if multiplier != 1 else val
                values[item_key] = val
                extraction_log.append(f"  [RETRY] {item_key} = {val:,.0f}")
        retry_count = sum(1 for v in values.values() if v is not None)
        if retry_count > success_count:
            logger.info("重试后提取到 %d/19 个财务数值 (+%d)",
                        retry_count, retry_count - success_count)

    # --- 最终清理：全页搜索后再次检查营收/成本比 ---
    _final_rev = values.get("营业收入")
    _final_cost = values.get("营业成本")
    if _final_rev and _final_cost and min(abs(_final_rev), abs(_final_cost)) > 0:
        _ratio = max(abs(_final_rev), abs(_final_cost)) / min(abs(_final_rev), abs(_final_cost))
        if _ratio > 50:
            _smaller = "营业成本" if abs(_final_cost) < abs(_final_rev) else "营业收入"
            logger.warning("最终清理: 营收与成本差%.0f倍，清除%s", _ratio, _smaller)
            values[_smaller] = None

    # ===================================================================
    # 第四步：Fallback 搜索 + 交叉验证 + 财务费用符号校正
    # ===================================================================

    # --- 4a) Fallback: 精确匹配未命中时用备选别名 ---
    for item_key, fb_aliases in ITEM_FALLBACK_ALIASES.items():
        if values.get(item_key) is not None:
            continue
        is_prior = item_key.startswith("期初")
        period = "prior" if is_prior else "current"
        actual_key = item_key.replace("期初", "") if is_prior else item_key
        income_items = {"营业收入", "营业成本", "净利润", "利润总额",
                        "财务费用", "所得税费用"}
        if actual_key in income_items:
            search_pages = inc_pages if inc_pages else all_stmt_pages
        else:
            search_pages = bs_pages if bs_pages else all_stmt_pages
        val = _search_value_in_pages(report, search_pages, fb_aliases, period)
        if val is not None:
            if multiplier != 1:
                val *= multiplier
            values[item_key] = val
            logger.info("  [FALLBACK] %s = %s (via %s)",
                        item_key, f"{val:,.0f}", fb_aliases[0])

    # --- 4b) 交叉验证：负债合计 >= 流动负债合计 ---
    _tl = values.get("负债合计")
    _cl = values.get("流动负债合计")
    if _tl is not None and _cl is not None and _tl > 0 and _cl > 0:
        if abs(_tl - _cl) / max(abs(_tl), abs(_cl)) < 0.001:
            logger.warning("负债合计(%.0f) ≈ 流动负债合计(%.0f)，可能遗漏非流动负债",
                           _tl, _cl)
            search_pg = bs_pages if bs_pages else all_stmt_pages
            for pg_num in search_pg:
                if pg_num < 1 or pg_num > len(report.pages):
                    continue
                text = report.pages[pg_num - 1].text or ""
                for line in text.split("\n"):
                    ls = line.strip()
                    item_name = _extract_item_name(ls)
                    if not item_name:
                        continue
                    clean_name = item_name.replace(" ", "")
                    if clean_name in ("负债合计", "负债总计", "负债总额"):
                        nums = _parse_numbers_from_line(ls, financial_only=True)
                        if not nums:
                            nums = _parse_numbers_from_line(ls)
                        v = _select_value(nums, "current")
                        if v is not None:
                            v_final = v * multiplier if multiplier != 1 else v
                            if v_final > _cl * 1.001:
                                logger.info("  [XVAL] 修正负债合计: %.0f -> %.0f",
                                            _tl, v_final)
                                values["负债合计"] = v_final
                                _tl = v_final
                            break

    # --- 4c) 交叉验证：会计恒等式 资产=负债+权益 ---
    _ta_v = values.get("资产总计")
    _tl_v = values.get("负债合计")
    _eq_v = values.get("所有者权益合计")
    if _ta_v and _tl_v and _eq_v:
        _id_diff = abs(_ta_v - _tl_v - _eq_v) / abs(_ta_v) if _ta_v else 0
        if _id_diff > 0.05:
            logger.warning("会计恒等式偏差 %.1f%%，尝试从合并报表重新提取",
                           _id_diff * 100)
            _recheck_pages = bs_pages if bs_pages else all_stmt_pages
            for _rk in ("资产总计", "负债合计"):
                _rv = _search_value_in_pages(
                    report, _recheck_pages, ITEM_ALIASES[_rk], "current")
                if _rv is not None:
                    _rv_final = _rv * multiplier if multiplier != 1 else _rv
                    if abs(_rv_final - values.get(_rk, 0)) > 1:
                        logger.info("  [XVAL] 重提取 %s: %.0f -> %.0f",
                                    _rk, values.get(_rk, 0), _rv_final)
                        values[_rk] = _rv_final

    # --- 4d) 交叉验证：毛利率合理性 ---
    _rev_f = values.get("营业收入")
    _cost_f = values.get("营业成本")
    if _rev_f and _cost_f and _rev_f > 0:
        _gm = (_rev_f - _cost_f) / _rev_f
        if _gm < -0.05:
            logger.warning("毛利率异常(%.1f%%)，营业成本可能取了营业总成本，"
                           "尝试重新精确提取", _gm * 100)
            _cost_pages = inc_pages if inc_pages else all_stmt_pages
            _new_cost = _search_value_in_pages(
                report, _cost_pages, ["营业成本"], "current")
            if _new_cost is not None:
                _nc = _new_cost * multiplier if multiplier != 1 else _new_cost
                _new_gm = (_rev_f - _nc) / _rev_f
                if -0.1 < _new_gm < 1.0 and _nc != _cost_f:
                    logger.info("  [XVAL] 修正营业成本: %.0f -> %.0f (毛利率 %.1f%%)",
                                _cost_f, _nc, _new_gm * 100)
                    values["营业成本"] = _nc

    # --- 4d-2) 交叉验证：成本远大于收入 → 营业收入可能取自母公司 ---
    _rev_f2 = values.get("营业收入")
    _cost_f2 = values.get("营业成本")
    if _rev_f2 and _cost_f2 and _cost_f2 > _rev_f2 * 2 and _rev_f2 > 0:
        logger.warning("营业成本(%.0f)是营业收入(%.0f)的%.1f倍，"
                       "营业收入可能取自母公司", _cost_f2, _rev_f2, _cost_f2 / _rev_f2)
        for pg_num in range(_section_start, len(report.pages) + 1):
            if _is_in_mother_section(report, pg_num):
                continue
            t = report.pages[pg_num - 1].text or ""
            head_t = "\n".join(t.split("\n")[:15])
            if not re.search(r"合\s*并\s*(?:及\s*公\s*司\s*)?利\s*润", head_t):
                continue
            if _count_data_lines(t) < 3:
                continue
            _nr = _search_value_in_pages(
                report, [pg_num], ITEM_ALIASES["营业收入"], "current",
                _allow_expand=False)
            if _nr is not None:
                _nrf = _nr * multiplier if multiplier != 1 else _nr
                if _nrf > _rev_f2 * 1.5:
                    _ngm = (_nrf - _cost_f2) / _nrf
                    if -0.1 < _ngm < 1.0:
                        logger.info("  [XVAL] 修正营业收入: %.0f -> %.0f (毛利率 %.1f%%)",
                                    _rev_f2, _nrf, _ngm * 100)
                        values["营业收入"] = _nrf
                        break

    # --- 4d-3) 交叉验证：净利率 > 毛利率 → 营业收入可能取自母公司 ---
    _rev_f3 = values.get("营业收入")
    _cost_f3 = values.get("营业成本")
    _ni_f3 = values.get("净利润")
    if _rev_f3 and _cost_f3 and _ni_f3 and _rev_f3 > 0:
        _gm3 = (_rev_f3 - _cost_f3) / _rev_f3
        _nm3 = _ni_f3 / _rev_f3
        if _nm3 > _gm3 + 0.05 and 0 < _gm3 < 0.9:
            logger.warning("净利率(%.1f%%) > 毛利率(%.1f%%)，"
                           "营业收入可能取自母公司", _nm3 * 100, _gm3 * 100)
            for pg_num in range(_section_start, len(report.pages) + 1):
                if _is_in_mother_section(report, pg_num):
                    continue
                t = report.pages[pg_num - 1].text or ""
                head_t = "\n".join(t.split("\n")[:15])
                if not re.search(
                        r"合\s*并\s*(?:及\s*公\s*司\s*)?利\s*润", head_t):
                    continue
                if _count_data_lines(t) < 3:
                    continue
                _nr3 = _search_value_in_pages(
                    report, [pg_num], ITEM_ALIASES["营业收入"], "current",
                    _allow_expand=False)
                if _nr3 is not None:
                    _nrf3 = _nr3 * multiplier if multiplier != 1 else _nr3
                    if _nrf3 > _rev_f3 * 1.1:
                        _ngm3 = (_nrf3 - _cost_f3) / _nrf3
                        _nnm3 = _ni_f3 / _nrf3
                        if _nnm3 <= _ngm3 + 0.05:
                            logger.info("  [XVAL] 修正营业收入: %.0f -> %.0f "
                                        "(毛利率 %.1f%%, 净利率 %.1f%%)",
                                        _rev_f3, _nrf3, _ngm3 * 100, _nnm3 * 100)
                            values["营业收入"] = _nrf3
                            break

    # --- 4e) 交叉验证：利润总额 - 所得税 ≈ 净利润 ---
    _pt = values.get("利润总额")
    _tax = values.get("所得税费用")
    _ni = values.get("净利润")
    if _pt and _tax and _ni:
        _expected_ni = _pt - _tax
        if abs(_ni) > 0 and abs(_ni - _expected_ni) / abs(_ni) > 0.15:
            logger.warning("净利润(%.0f) != 利润总额(%.0f) - 所得税(%.0f) = %.0f，"
                           "偏差 %.1f%%，可能取自母公司利润表",
                           _ni, _pt, _tax, _expected_ni,
                           abs(_ni - _expected_ni) / abs(_ni) * 100)
            if abs(_expected_ni) > abs(_ni) * 0.5:
                logger.info("  [XVAL] 修正净利润: %.0f -> %.0f", _ni, _expected_ni)
                values["净利润"] = _expected_ni

    # --- 最终会计恒等式强化校验（LLM辅助修复合并/母公司混淆） ---
    _final_ta = values.get("资产总计")
    _final_tl = values.get("负债合计")
    _final_eq = values.get("所有者权益合计")
    if _final_ta and _final_tl and _final_eq and _final_ta > 0:
        _final_diff = abs(_final_ta - _final_tl - _final_eq) / abs(_final_ta)
        if _final_diff > 0.10:
            logger.warning(
                "会计恒等式严重偏差 %.1f%%，疑似合并/母公司数据混淆，启动LLM修复",
                _final_diff * 100)

            _anchor_pg = None
            _search_scope = bs_pages if bs_pages else all_stmt_pages
            for pg in _search_scope:
                if pg < 1 or pg > len(report.pages):
                    continue
                if "资产总计" in (report.pages[pg - 1].text or ""):
                    _anchor_pg = pg
                    break

            if _anchor_pg is not None:
                _repair_pages = list(range(
                    max(1, _anchor_pg - 2),
                    min(len(report.pages) + 1, _anchor_pg + 8),
                ))
                _repair_bs_text = []
                for pg in _repair_pages:
                    t = report.pages[pg - 1].text or ""
                    if t.strip():
                        _repair_bs_text.append(f"--- 第{pg}页 ---\n{t}")

                _repair_prompt = (
                    "你是上市公司年报合并资产负债表数据提取专家。\n"
                    "以下页面包含合并资产负债表和可能的母公司资产负债表。\n"
                    "请【只】从【合并资产负债表】中提取以下数据，忽略母公司报表。\n"
                    "判断依据：合并报表通常标题含'合并资产负债表'，且数值远大于母公司。\n\n"
                    f"已知合并报表资产总计 = {_final_ta:,.0f} 元\n"
                    "请提取的负债合计和所有者权益合计之和应接近此资产总计。\n\n"
                    + "\n".join(_repair_bs_text[:10])[:8000]
                    + "\n\n返回纯JSON：{\"负债合计\": 数值, \"所有者权益合计\": 数值}\n"
                    "金额单位为元，缺失填null。"
                )

                try:
                    _client = get_client_for_provider(llm_provider)
                    _model = llm_model or get_model_for_provider(llm_provider)
                    _raw = chat_completion(
                        _client, _model,
                        [{"role": "user", "content": _repair_prompt}],
                        temperature=0.0,
                    )
                    _text = _raw.strip()
                    _md = re.search(r"```(?:json)?\s*\n?(.*?)```", _text, re.DOTALL)
                    if _md:
                        _text = _md.group(1).strip()
                    _jm = re.search(r"\{[\s\S]*\}", _text)
                    if _jm:
                        _repair_data = json.loads(_jm.group())
                        _r_tl = _repair_data.get("负债合计")
                        _r_eq = _repair_data.get("所有者权益合计")
                        if _r_tl is not None:
                            _r_tl = float(_r_tl)
                        if _r_eq is not None:
                            _r_eq = float(_r_eq)

                        if _r_tl and _r_eq and _r_tl > 0 and _r_eq > 0:
                            _repair_diff = abs(
                                _final_ta - _r_tl - _r_eq) / abs(_final_ta)
                            if _repair_diff < _final_diff * 0.5:
                                logger.info(
                                    "  [LLM恒等式修复] 负债合计: %s -> %s",
                                    f"{_final_tl:,.0f}", f"{_r_tl:,.0f}")
                                logger.info(
                                    "  [LLM恒等式修复] 所有者权益合计: %s -> %s",
                                    f"{_final_eq:,.0f}", f"{_r_eq:,.0f}")
                                logger.info(
                                    "  [LLM恒等式修复] 偏差 %.1f%% -> %.1f%%",
                                    _final_diff * 100, _repair_diff * 100)
                                values["负债合计"] = _r_tl
                                values["所有者权益合计"] = _r_eq
                            else:
                                logger.warning(
                                    "  [LLM恒等式修复] LLM偏差 %.1f%% 未改善",
                                    _repair_diff * 100)
                except Exception as _exc:
                    logger.warning("  [LLM恒等式修复] 失败: %s", _exc)

            _repair_ta = values.get("资产总计")
            _repair_tl = values.get("负债合计")
            _repair_eq = values.get("所有者权益合计")
            if _repair_ta and _repair_tl and _repair_eq:
                _still_diff = abs(
                    _repair_ta - _repair_tl - _repair_eq) / abs(_repair_ta)
                if _still_diff > 0.05 and _repair_tl > 0:
                    _derived = _repair_ta - _repair_tl
                    if _derived > 0:
                        logger.info(
                            "  [恒等式推导] 所有者权益合计 = %s (资产-负债)",
                            f"{_derived:,.0f}")
                        values["所有者权益合计"] = _derived

    return values


# ---------------------------------------------------------------------------
# Agent 节点函数
# ---------------------------------------------------------------------------

def document_parser_node(state: PipelineState) -> dict:
    pdf_path: str = state["pdf_path"]
    errors: list[str] = list(state.get("errors", []))
    logger.info("开始解析: %s", pdf_path)

    parsed_report: ParsedReport | None = None
    financial_tables: dict[str, pd.DataFrame] = {}
    extracted_values: dict[str, float | None] = {}
    vs: VectorStore | None = None

    try:
        if pdf_path in _preparse_cache:
            parsed_report = _preparse_cache[pdf_path]
            logger.info("使用预解析缓存: %d页, %d个表格",
                         parsed_report.total_pages, len(parsed_report.tables))
        else:
            parsed_report = parse_annual_report(pdf_path)
            logger.info("PDF解析完成: %d页, %d个表格",
                         parsed_report.total_pages, len(parsed_report.tables))
    except Exception as exc:
        msg = f"PDF解析失败: {exc}"
        logger.error(msg)
        errors.append(msg)
        return {"parsed_report": None, "financial_tables": {},
                "extracted_values": {}, "vector_store": None, "errors": errors}

    try:
        financial_tables = extract_financial_data(parsed_report)
        for stmt_type, df in financial_tables.items():
            logger.info("识别到 %s: %d 行", stmt_type, len(df))
    except Exception as exc:
        logger.warning("财务报表DataFrame提取失败: %s", exc)

    _provider = state.get("llm_provider", "tengri")
    _llm_model = state.get("llm_model", "")
    extracted_values = _extract_key_values(
        parsed_report, llm_provider=_provider, llm_model=_llm_model,
    )
    success_count = sum(1 for v in extracted_values.values() if v is not None)
    logger.info("提取到 %d/%d 个财务数值", success_count, len(extracted_values))
    if success_count == 0:
        errors.append("未能从年报中提取到任何财务数值")

    try:
        pages_data = get_pages_text_with_metadata(parsed_report)
        chunker = TextChunker()
        chunks = chunker.chunk_by_pages(pages_data)
        if chunks:
            vs = VectorStore()
            vs.build_index(chunks)
            logger.info("向量索引构建完成: %d 个分块", vs.size)
        else:
            errors.append("文本分块为空，无法构建向量索引")
    except Exception as exc:
        msg = f"向量索引构建失败: {exc}"
        logger.error(msg)
        errors.append(msg)

    return {
        "parsed_report": parsed_report,
        "financial_tables": financial_tables,
        "extracted_values": extracted_values,
        "vector_store": vs,
        "errors": errors,
    }
