"""PDF 解析工具 -- 从年报PDF中提取文本和表格"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import pdfplumber
import pandas as pd


@dataclass
class PageContent:
    """单页内容"""
    page_number: int
    text: str
    tables: list[list[list[str]]]
    section_titles: list[str] = field(default_factory=list)


@dataclass
class TableData:
    """提取的表格数据"""
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    table_type: str = ""
    section_title: str = ""


@dataclass
class ParsedReport:
    """解析后的年报数据"""
    file_path: str
    total_pages: int
    pages: list[PageContent]
    tables: list[TableData]
    financial_statements: dict[str, list[TableData]]
    full_text: str


SECTION_PATTERNS = [
    re.compile(r'^第[一二三四五六七八九十]+[节章]\s+.+'),
    re.compile(r'^[一二三四五六七八九十]+[、.]\s*.+'),
    re.compile(r'^\d+[、.]\s*.+'),
    re.compile(r'^（[一二三四五六七八九十]+）\s*.+'),
    re.compile(r'^[IVX]+[、.]\s*.+'),
]

FINANCIAL_STATEMENT_KEYWORDS = {
    "资产负债表": ["资产负债表", "合并资产负债表", "母公司资产负债表"],
    "利润表": ["利润表", "合并利润表", "合并利润及其他综合收益表", "母公司利润表"],
    "现金流量表": ["现金流量表", "合并现金流量表", "母公司现金流量表"],
}


def _identify_sections(text: str) -> list[str]:
    """从页面文本中识别章节标题"""
    titles = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or len(line) > 60:
            continue
        for pattern in SECTION_PATTERNS:
            if pattern.match(line):
                titles.append(line)
                break
    return titles


def _identify_table_type(headers: list[str], page_text: str) -> str:
    """识别表格是否为财务报表及其类型"""
    combined = " ".join(h for h in headers if h)
    search_text = combined + " " + page_text[:500]

    for stmt_type, keywords in FINANCIAL_STATEMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in search_text:
                return stmt_type
    return ""


def _clean_table(raw_table: list[list]) -> tuple[list[str], list[list[str]]]:
    """清洗原始表格数据，分离表头和数据行"""
    if not raw_table or len(raw_table) < 2:
        return [], []

    headers = [str(cell).strip() if cell else "" for cell in raw_table[0]]
    rows = []
    for row in raw_table[1:]:
        cleaned = [str(cell).strip() if cell else "" for cell in row]
        if any(c for c in cleaned):
            rows.append(cleaned)
    return headers, rows


def parse_annual_report(pdf_path: str | Path) -> ParsedReport:
    """
    解析年报PDF，提取全部文本和表格。

    返回 ParsedReport，包含逐页内容、所有表格（含元数据）、
    识别出的三大财务报表以及全文拼接文本。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"文件不存在: {pdf_path}")

    pages: list[PageContent] = []
    all_tables: list[TableData] = []
    financial_statements: dict[str, list[TableData]] = {}
    text_parts: list[str] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        current_section = ""

        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            text = page.extract_text() or ""
            raw_tables = page.extract_tables() or []

            section_titles = _identify_sections(text)
            if section_titles:
                current_section = section_titles[-1]

            page_tables_raw: list[list[list[str]]] = []
            for t_idx, rt in enumerate(raw_tables):
                if not rt:
                    continue

                cleaned = [
                    [str(c).strip() if c else "" for c in row]
                    for row in rt
                ]
                page_tables_raw.append(cleaned)

                headers, rows = _clean_table(rt)
                table_type = _identify_table_type(headers, text)

                td = TableData(
                    page_number=page_num,
                    table_index=t_idx,
                    headers=headers,
                    rows=rows,
                    table_type=table_type,
                    section_title=current_section,
                )
                all_tables.append(td)

                if table_type:
                    financial_statements.setdefault(table_type, []).append(td)

            pages.append(PageContent(
                page_number=page_num,
                text=text,
                tables=page_tables_raw,
                section_titles=section_titles,
            ))
            text_parts.append(text)

    return ParsedReport(
        file_path=str(pdf_path),
        total_pages=total_pages,
        pages=pages,
        tables=all_tables,
        financial_statements=financial_statements,
        full_text="\n\n".join(text_parts),
    )


def table_to_dataframe(table_data: TableData) -> pd.DataFrame:
    """将 TableData 转换为 pandas DataFrame，自动处理重复列名"""
    if not table_data.headers or not table_data.rows:
        return pd.DataFrame()

    headers = list(table_data.headers)
    seen: dict[str, int] = {}
    for i, h in enumerate(headers):
        if h in seen:
            seen[h] += 1
            headers[i] = f"{h}_{seen[h]}"
        else:
            seen[h] = 0

    max_cols = len(headers)
    rows = []
    for row in table_data.rows:
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)


def extract_financial_data(report: ParsedReport) -> dict[str, pd.DataFrame]:
    """
    从解析结果中提取三大财务报表，返回 DataFrame 字典。

    跨页的同类型表格会自动拼接。
    """
    result: dict[str, pd.DataFrame] = {}
    for stmt_type, tables in report.financial_statements.items():
        frames = []
        for td in tables:
            df = table_to_dataframe(td)
            if not df.empty:
                frames.append(df)
        if frames:
            result[stmt_type] = pd.concat(frames, ignore_index=True)
    return result


def get_pages_text_with_metadata(report: ParsedReport) -> list[dict]:
    """
    将每页文本与元数据组合，供后续向量化使用。

    每项包含 page_number、text、section_titles、has_tables、table_types。
    """
    table_type_map: dict[int, list[str]] = {}
    for t in report.tables:
        if t.table_type:
            table_type_map.setdefault(t.page_number, []).append(t.table_type)

    results = []
    for page in report.pages:
        results.append({
            "page_number": page.page_number,
            "text": page.text,
            "section_titles": page.section_titles,
            "has_tables": len(page.tables) > 0,
            "table_types": table_type_map.get(page.page_number, []),
        })
    return results
