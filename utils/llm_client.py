"""LLM 客户端封装 -- Tengri API 调用接口

使用全局单例避免多线程同时初始化 OpenAI 客户端时的导入锁死锁。
所有 API 调用对 403 / 429 / 5xx 自动重试（固定 0.5s 间隔）。
SDK 内置重试已禁用（max_retries=0），仅使用本模块的重试逻辑。
"""

import logging
import random
import threading
import time

from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
from config import TengriConfig

logger = logging.getLogger(__name__)

_tengri_client: OpenAI | None = None
_lock = threading.Lock()

_RETRYABLE_CODES = {400, 403, 429, 500, 502, 503, 504}
_MAX_RETRIES = 10
_RETRY_DELAY_BASE = 1.0
_RETRY_DELAY_MAX = 30.0


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_CODES:
        return True
    return False


def _backoff_delay(attempt: int) -> float:
    """指数退避延迟: 1s, 2s, 4s, 8s ... 上限 30s，附加抖动。"""
    delay = min(_RETRY_DELAY_BASE * (2 ** attempt), _RETRY_DELAY_MAX)
    return delay + random.uniform(0, delay * 0.1)


def get_tengri_client() -> OpenAI:
    """获取讯蒙科技 Tengri LLM 客户端（线程安全单例）"""
    global _tengri_client
    if _tengri_client is None:
        with _lock:
            if _tengri_client is None:
                _tengri_client = OpenAI(
                    api_key=TengriConfig.API_KEY,
                    base_url=TengriConfig.BASE_URL,
                    timeout=TengriConfig.TIMEOUT,
                    max_retries=0,
                )
    return _tengri_client


def warmup():
    """在主线程中预初始化客户端，避免多线程首次导入死锁。"""
    get_tengri_client()


class EmptyResponseError(Exception):
    """LLM 返回了空内容（None 或空字符串）。"""


def chat_completion(client: OpenAI, model: str, messages: list, **kwargs) -> str:
    """统一的 Chat Completion 调用（带重试）。

    对 HTTP 5xx / 429 / 403 自动重试；对空响应内容也触发重试。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "chat_completion 重试 %d/%d (响应内容为空, "
                        "finish_reason=%s, model=%s)",
                        attempt + 1, _MAX_RETRIES,
                        getattr(response.choices[0], 'finish_reason', '?'),
                        model,
                    )
                    time.sleep(_backoff_delay(attempt))
                    continue
                raise EmptyResponseError(
                    f"LLM 连续 {_MAX_RETRIES} 次返回空内容 "
                    f"(finish_reason={getattr(response.choices[0], 'finish_reason', '?')})"
                )
            return content
        except EmptyResponseError:
            raise
        except Exception as exc:
            if _should_retry(exc) and attempt < _MAX_RETRIES - 1:
                if isinstance(exc, (APITimeoutError, APIConnectionError)):
                    reason = type(exc).__name__
                else:
                    status = getattr(exc, 'status_code', '?')
                    body_preview = ""
                    if hasattr(exc, 'body'):
                        body_preview = f" body={str(exc.body)[:200]}"
                    elif hasattr(exc, 'message'):
                        body_preview = f" msg={str(exc.message)[:200]}"
                    reason = f"HTTP {status}{body_preview}"
                delay = _backoff_delay(attempt)
                logger.warning(
                    "chat_completion 重试 %d/%d (%s, %.1fs后重试)",
                    attempt + 1, _MAX_RETRIES, reason, delay,
                )
                time.sleep(delay)
                continue
            raise


def get_client_for_provider(provider: str = "tengri") -> OpenAI:
    """根据 provider 返回对应的 LLM 客户端。"""
    return get_tengri_client()


def get_model_for_provider(provider: str = "tengri") -> str:
    """根据 provider 返回对应的模型名称。"""
    return TengriConfig.MODEL


def get_embedding(client: OpenAI, model: str, text: str) -> list[float]:
    """获取单条文本的向量嵌入（带重试）"""
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            if _should_retry(exc) and attempt < _MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                logger.warning("get_embedding 重试 %d/%d (HTTP %s, %.1fs后重试)",
                               attempt + 1, _MAX_RETRIES,
                               getattr(exc, 'status_code', '?'), delay)
                time.sleep(delay)
                continue
            raise


def get_embeddings_batch(
    client: OpenAI,
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """批量获取多条文本的向量嵌入（带重试）。

    OpenAI-compatible API 支持 input 为字符串列表，一次请求返回多个嵌入向量。
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=model,
                input=texts,
            )
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]
        except Exception as exc:
            if _should_retry(exc) and attempt < _MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                logger.warning("get_embeddings_batch 重试 %d/%d (HTTP %s, %.1fs后重试)",
                               attempt + 1, _MAX_RETRIES,
                               getattr(exc, 'status_code', '?'), delay)
                time.sleep(delay)
                continue
            raise
