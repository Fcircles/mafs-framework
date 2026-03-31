"""基线4: 通用RAG方法 -- LangChain+FAISS，不融入财务模型计算逻辑"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from baselines.base import BaseBaseline
from config import TengriConfig, RAGConfig
from models import AnalysisResult, IndicatorResult
from utils.llm_client import get_tengri_client, chat_completion
from utils.pdf_parser import parse_annual_report

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 预定义检索查询（每个覆盖一个指标类别）
# ---------------------------------------------------------------------------

def _build_rag_queries(industry: str) -> list[dict[str, str]]:
    """根据行业构建带有 Z-Score 模型版本说明的 RAG 查询列表。"""
    if industry == "制造业":
        zscore_query = (
            "该公司的流动资产、流动负债、总资产、总负债、营运资本、留存收益、"
            "息税前利润、所有者权益、营业收入是多少？"
            "请计算制造业Altman Z-Score（Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5，"
            "其中X4=账面权益/总负债）。>2.99安全区，1.81~2.99灰色区，<1.81危险区。"
        )
    else:
        zscore_query = (
            "该公司的流动资产、流动负债、总资产、总负债、营运资本、留存收益、"
            "息税前利润、所有者权益是多少？"
            "请计算非制造业Altman Z''-Score（Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4，"
            "其中X4=账面权益/总负债）。>2.60安全区，1.10~2.60灰色区，<1.10危险区。"
        )
    return [
        {
            "query": "该公司的净利润、营业收入、总资产、所有者权益是多少？请计算ROE并做杜邦分解",
            "category": "杜邦分析",
        },
        {
            "query": zscore_query,
            "category": "Z-Score",
        },
        {
            "query": "该公司的流动资产、流动负债、存货、总资产、总负债分别是多少？请分析偿债能力",
            "category": "偿债能力",
        },
        {
            "query": "该公司的营业收入、营业成本、净利润、总资产、所有者权益是多少？请分析盈利能力",
            "category": "盈利能力",
        },
        {
            "query": "该公司的营业收入、营业成本、应收账款、存货、总资产分别是多少？请分析营运能力",
            "category": "营运能力",
        },
    ]

_SYNTHESIS_QUERY = "请根据以上分析，给出200-400字的综合财务解读"

# ---------------------------------------------------------------------------
# RAG 回答的 Prompt
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT = """\
你是一位专业的财务分析师。你将收到从年报中检索到的相关段落，请基于这些段落回答用户的问题。
如果能从文本中提取到具体数值，请计算相应的财务指标。
请以JSON格式返回结果，不要添加其他内容。"""

_RAG_ANSWER_TEMPLATE = """\
以下是从{company_name}（{company_code}）{year}年年报中检索到的相关内容：

{context}

问题：{query}

请以JSON格式返回（不要输出其他内容）：
{{
  "indicators": [
    {{"name": "指标名称", "value": 数值或null, "formula": "计算公式", "source_text": "依据的原文内容"}}
  ],
  "analysis": "基于检索内容的分析说明"
}}"""

_SYNTHESIS_PROMPT = """\
你是一位专业的财务分析师。以下是对{company_name}（{company_code}）{year}年年报的分项分析结果：

{section_analyses}

请综合以上分析，给出200-400字的整体财务状况评价，涵盖盈利能力、偿债能力、营运效率和风险提示。
直接输出分析文本，不要使用JSON格式。"""


class GeneralRAGBaseline(BaseBaseline):
    """基线4: 通用RAG -- LangChain+FAISS，不融入财务模型计算逻辑"""

    def __init__(self, top_k: int | None = None):
        self._top_k = top_k or RAGConfig.TOP_K

    @property
    def name(self) -> str:
        return "general_rag"

    def analyze(
        self,
        pdf_path: str | Path,
        company_code: str,
        company_name: str,
        year: int,
        industry: str,
    ) -> AnalysisResult:
        pdf_path = Path(pdf_path)

        vectorstore = self._build_vectorstore(pdf_path)
        client = get_tengri_client()

        all_indicators: list[IndicatorResult] = []
        section_texts: list[str] = []

        rag_queries = _build_rag_queries(industry)
        for q_info in rag_queries:
            query = q_info["query"]
            category = q_info["category"]

            docs = self._search_with_retry(vectorstore, query)
            context = "\n\n".join(
                f"[段落{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
            )

            source_pages = [
                doc.metadata.get("page_number")
                for doc in docs
                if doc.metadata.get("page_number") is not None
            ]

            prompt = _RAG_ANSWER_TEMPLATE.format(
                company_name=company_name,
                company_code=company_code,
                year=year,
                query=query,
                context=context,
            )

            raw_answer = chat_completion(
                client, TengriConfig.MODEL,
                [
                    {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            parsed = self._parse_answer(raw_answer)
            section_texts.append(
                f"【{category}】\n{parsed.get('analysis', '')}"
            )

            for item in parsed.get("indicators", []):
                val = item.get("value")
                if val is not None:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = None
                all_indicators.append(IndicatorResult(
                    name=item.get("name", ""),
                    value=val,
                    formula=item.get("formula", ""),
                    source_page=source_pages[0] if source_pages else None,
                    source_text=item.get("source_text") or "",
                    category=category,
                ))

        interpretation = self._synthesize(
            client, section_texts,
            company_name, company_code, year,
        )

        return AnalysisResult(
            company_code=company_code,
            company_name=company_name,
            year=year,
            industry=industry,
            method=self.name,
            indicators=all_indicators,
            interpretation=interpretation,
            metadata={
                "model": TengriConfig.MODEL,
                "top_k": self._top_k,
                "rag_queries_count": len(rag_queries),
            },
        )

    # ------------------------------------------------------------------
    # 带重试的相似度检索
    # ------------------------------------------------------------------

    def _search_with_retry(
        self, vectorstore: FAISS, query: str, max_retries: int = 5,
    ) -> list:
        for attempt in range(max_retries):
            try:
                return vectorstore.similarity_search(query, k=self._top_k)
            except Exception as exc:
                if attempt < max_retries - 1:
                    delay = min(2 ** attempt + 1, 30)
                    logger.warning(
                        "similarity_search 重试 %d/%d (%s, %.0fs后重试)",
                        attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                raise

    # ------------------------------------------------------------------
    # 向量库构建
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vectorstore(pdf_path: Path) -> FAISS:
        """用手动分批嵌入构建 FAISS 向量库，跳过失败批次避免长度不匹配"""
        report = parse_annual_report(pdf_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        )

        docs_with_meta: list[dict] = []
        for page in report.pages:
            text = page.text.strip()
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                docs_with_meta.append({
                    "text": chunk,
                    "metadata": {"page_number": page.page_number},
                })

        from utils.llm_client import get_tengri_client, get_embeddings_batch

        client = get_tengri_client()
        batch_size = 16
        ok_texts: list[str] = []
        ok_metas: list[dict] = []
        ok_embeds: list[list[float]] = []

        for i in range(0, len(docs_with_meta), batch_size):
            batch = docs_with_meta[i:i + batch_size]
            batch_texts = [d["text"] for d in batch]
            try:
                vecs = get_embeddings_batch(client, TengriConfig.EMBEDDING_MODEL, batch_texts)
                if len(vecs) == len(batch_texts):
                    ok_texts.extend(batch_texts)
                    ok_metas.extend(d["metadata"] for d in batch)
                    ok_embeds.extend(vecs)
                else:
                    logger.warning("嵌入批次 %d-%d 返回长度不匹配 (%d/%d)，跳过",
                                   i, i + len(batch), len(vecs), len(batch_texts))
            except Exception as exc:
                logger.warning("嵌入批次 %d-%d 失败: %s，跳过", i, i + len(batch), exc)

        if not ok_texts:
            raise ValueError("所有嵌入批次均失败，无法构建向量库")

        logger.info("嵌入完成: %d/%d 文本块成功", len(ok_texts), len(docs_with_meta))
        text_embedding_pairs = list(zip(ok_texts, ok_embeds))
        return FAISS.from_embeddings(text_embedding_pairs, OpenAIEmbeddings(
            model=TengriConfig.EMBEDDING_MODEL,
            api_key=TengriConfig.API_KEY,
            base_url=TengriConfig.BASE_URL,
            max_retries=8,
        ), metadatas=ok_metas)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_answer(raw_text: str) -> dict:
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
        return {"indicators": [], "analysis": raw_text}

    @staticmethod
    def _synthesize(
        client,
        section_texts: list[str],
        company_name: str,
        company_code: str,
        year: int,
    ) -> str:
        prompt = _SYNTHESIS_PROMPT.format(
            company_name=company_name,
            company_code=company_code,
            year=year,
            section_analyses="\n\n".join(section_texts),
        )
        return chat_completion(
            client, TengriConfig.MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
