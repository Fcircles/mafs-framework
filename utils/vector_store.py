"""向量存储与检索 -- 基于FAISS的向量库管理"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import faiss

from config import TengriConfig, RAGConfig, ConcurrencyConfig
from utils.llm_client import get_tengri_client, get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """文本分块，携带元数据"""
    text: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """文本分块器，支持按页面或章节边界切分"""

    def __init__(self, chunk_size: int = RAGConfig.CHUNK_SIZE,
                 chunk_overlap: int = RAGConfig.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_by_pages(self, pages_data: list[dict]) -> list[TextChunk]:
        """
        按页面切分，超长页面进一步拆分。

        参数 pages_data 为 get_pages_text_with_metadata() 的返回值。
        """
        chunks: list[TextChunk] = []
        for page in pages_data:
            text = page["text"].strip()
            if not text:
                continue

            base_meta = {
                "page_number": page["page_number"],
                "section_titles": page.get("section_titles", []),
                "has_tables": page.get("has_tables", False),
                "table_types": page.get("table_types", []),
                "chunk_type": "page",
            }

            if len(text) <= self.chunk_size:
                chunks.append(TextChunk(text=text, metadata=base_meta.copy()))
            else:
                for i, sub in enumerate(self._split_text(text)):
                    meta = base_meta.copy()
                    meta["sub_chunk_index"] = i
                    chunks.append(TextChunk(text=sub, metadata=meta))
        return chunks

    def chunk_by_sections(self, full_text: str,
                          section_boundaries: list[dict] | None = None) -> list[TextChunk]:
        """
        按章节标题切分。

        section_boundaries 每项含 title 和 start_pos。
        若为 None 则退化为普通段落切分。
        """
        if not section_boundaries:
            return [TextChunk(text=t, metadata={"chunk_type": "text"})
                    for t in self._split_text(full_text)]

        chunks: list[TextChunk] = []
        n = len(section_boundaries)
        for i, boundary in enumerate(section_boundaries):
            start = boundary.get("start_pos", 0)
            end = section_boundaries[i + 1]["start_pos"] if i + 1 < n else len(full_text)
            section_text = full_text[start:end].strip()
            if not section_text:
                continue

            meta = {
                "section_title": boundary.get("title", ""),
                "chunk_type": "section",
            }

            if len(section_text) <= self.chunk_size:
                chunks.append(TextChunk(text=section_text, metadata=meta.copy()))
            else:
                for j, sub in enumerate(self._split_text(section_text)):
                    m = meta.copy()
                    m["sub_chunk_index"] = j
                    chunks.append(TextChunk(text=sub, metadata=m))
        return chunks

    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[str]:
        """将长文本按段落/句子边界拆分"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        paragraphs = re.split(r'\n\s*\n', text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if not current:
                if len(para) <= self.chunk_size:
                    current = para
                else:
                    sub = self._split_long_paragraph(para)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                continue

            if len(current) + len(para) + 1 <= self.chunk_size:
                current = current + "\n" + para
            else:
                chunks.append(current)
                if len(para) <= self.chunk_size:
                    current = para
                else:
                    sub = self._split_long_paragraph(para)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""

        if current.strip():
            chunks.append(current)

        return [c for c in chunks if c.strip()]

    def _split_long_paragraph(self, text: str) -> list[str]:
        """按句子边界拆分超长段落"""
        sentences = re.split(r'(?<=[。！？；\n])', text)
        chunks: list[str] = []
        current = ""

        for sent in sentences:
            if not sent.strip():
                continue
            if not current:
                current = sent
            elif len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                chunks.append(current)
                current = sent

        if current.strip():
            chunks.append(current)

        return chunks if chunks else [text[:self.chunk_size]]


class VectorStore:
    """基于 FAISS 的向量存储与检索"""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[TextChunk] = []
        self._client = None
        self._embedding_dim: int | None = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_tengri_client()
        return self._client

    def _get_embedding(self, text: str) -> list[float]:
        return get_embedding(self.client, TengriConfig.EMBEDDING_MODEL, text)

    def _get_embeddings_batch(self, texts: list[str],
                              batch_size: int | None = None) -> np.ndarray:
        """批量获取向量嵌入，返回归一化后的 numpy 数组。

        使用批量API（每次请求发送 batch_size 条文本）+ 多线程并发
        同时发送多个批次请求，大幅缩短向量化耗时。
        """
        batch_size = batch_size or ConcurrencyConfig.EMBEDDING_BATCH_SIZE
        max_workers = ConcurrencyConfig.EMBEDDING_MAX_WORKERS

        batches = [
            (i, texts[i:i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]

        results: dict[int, list[list[float]]] = {}

        def _embed_one_batch(batch_info: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
            idx, batch_texts = batch_info
            embs = get_embeddings_batch(
                self.client, TengriConfig.EMBEDDING_MODEL, batch_texts,
            )
            return idx, embs

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_embed_one_batch, b): b[0]
                for b in batches
            }
            for future in as_completed(futures):
                idx, embs = future.result()
                results[idx] = embs

        all_emb: list[list[float]] = []
        for start_idx, _ in batches:
            all_emb.extend(results[start_idx])

        logger.info("向量嵌入完成: %d 条文本, %d 批次, %d 并发",
                     len(texts), len(batches), max_workers)
        vectors = np.array(all_emb, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def build_index(self, chunks: list[TextChunk]) -> None:
        """从文本分块构建 FAISS 内积索引"""
        if not chunks:
            raise ValueError("分块列表为空")

        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = self._get_embeddings_batch(texts)

        self._embedding_dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self._embedding_dim)
        self.index.add(vectors)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        语义检索，返回最相关的文本分块。

        每项包含 text、score、metadata。
        """
        if self.index is None or not self.chunks:
            raise RuntimeError("索引未构建，请先调用 build_index()")

        if top_k is None:
            top_k = RAGConfig.TOP_K

        query_vec = np.array([self._get_embedding(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk.text,
                "score": float(score),
                "metadata": chunk.metadata,
            })
        return results

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, save_dir: str | Path) -> None:
        """将索引和元数据保存到磁盘"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / "index.faiss"))

        metadata = [{"text": c.text, "metadata": c.metadata} for c in self.chunks]
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: str | Path) -> None:
        """从磁盘加载索引和元数据"""
        save_dir = Path(save_dir)

        index_path = save_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        self.index = faiss.read_index(str(index_path))
        self._embedding_dim = self.index.d

        meta_path = save_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [
            TextChunk(text=item["text"], metadata=item["metadata"])
            for item in data
        ]

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
