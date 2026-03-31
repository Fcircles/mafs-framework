"""基线方法抽象基类"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from models import AnalysisResult

logger = logging.getLogger(__name__)


class BaseBaseline(ABC):
    """所有基线方法的抽象基类，定义统一接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """方法标识名称，用于 AnalysisResult.method 字段"""
        ...

    @abstractmethod
    def analyze(
        self,
        pdf_path: str | Path,
        company_code: str,
        company_name: str,
        year: int,
        industry: str,
    ) -> AnalysisResult:
        """
        对单份年报执行分析。

        Parameters
        ----------
        pdf_path : 年报PDF文件路径
        company_code : 股票代码
        company_name : 公司简称
        year : 年报年份
        industry : 所属行业

        Returns
        -------
        AnalysisResult
        """
        ...

    def analyze_batch(
        self,
        tasks: list[dict],
        *,
        stop_on_error: bool = False,
    ) -> list[AnalysisResult]:
        """
        批量分析多份年报。

        Parameters
        ----------
        tasks : 每项为 dict，包含 analyze() 所需的全部参数
        stop_on_error : 遇到错误时是否中止（默认跳过并记录）

        Returns
        -------
        成功分析的结果列表
        """
        results: list[AnalysisResult] = []
        for i, task in enumerate(tasks, 1):
            key = f"{task.get('company_name', '?')}_{task.get('year', '?')}"
            logger.info("[%s] (%d/%d) 开始分析 %s", self.name, i, len(tasks), key)
            t0 = time.time()
            try:
                result = self.analyze(**task)
                elapsed = time.time() - t0
                result.metadata["elapsed_seconds"] = round(elapsed, 2)
                results.append(result)
                logger.info("[%s] (%d/%d) %s 完成，耗时 %.1fs",
                            self.name, i, len(tasks), key, elapsed)
            except Exception:
                logger.exception("[%s] (%d/%d) %s 分析失败", self.name, i, len(tasks), key)
                if stop_on_error:
                    raise
        return results
