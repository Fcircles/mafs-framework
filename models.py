"""统一输出数据模型 -- 供基线方法、Agent框架、评价指标模块共享"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class IndicatorResult:
    """单个财务指标计算结果"""

    name: str
    value: float | None
    formula: str
    source_page: int | None
    source_text: str
    category: str  # "杜邦分析"/"偿债能力"/"盈利能力"/"营运能力"/"Z-Score"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IndicatorResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AnalysisResult:
    """年报分析结果（统一输出格式）"""

    company_code: str
    company_name: str
    year: int
    industry: str
    method: str
    indicators: list[IndicatorResult]
    interpretation: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalysisResult:
        indicators = [
            IndicatorResult.from_dict(ind) if isinstance(ind, dict) else ind
            for ind in d.get("indicators", [])
        ]
        return cls(
            company_code=d["company_code"],
            company_name=d["company_name"],
            year=d["year"],
            industry=d["industry"],
            method=d["method"],
            indicators=indicators,
            interpretation=d.get("interpretation", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, s: str) -> AnalysisResult:
        return cls.from_dict(json.loads(s))

    def get_indicator(self, name: str) -> IndicatorResult | None:
        for ind in self.indicators:
            if ind.name == name:
                return ind
        return None

    def get_indicators_by_category(self, category: str) -> list[IndicatorResult]:
        return [ind for ind in self.indicators if ind.category == category]

    @classmethod
    def from_pipeline_output(cls, result: dict[str, Any]) -> AnalysisResult:
        """将 orchestrator 管道输出的嵌套 dict 转换为 AnalysisResult。

        管道输出格式:
            indicators: {"杜邦分析": {"ROE": 0.15, ...}, "Z-Score": {...}, ...}
            evidence_map: [{"indicator_name": ..., "page_number": ..., ...}, ...]
        """
        raw_indicators = result.get("indicators", {})
        evidence_map = result.get("evidence_map", [])

        evidence_by_name: dict[str, dict] = {}
        for ev in evidence_map:
            name = ev.get("indicator_name", "")
            if name not in evidence_by_name:
                evidence_by_name[name] = ev

        indicator_list: list[IndicatorResult] = []
        for category, data in raw_indicators.items():
            if category == "计算缺失项" or not isinstance(data, dict):
                continue
            for name, value in data.items():
                ev = evidence_by_name.get(name, {})
                indicator_list.append(IndicatorResult(
                    name=name,
                    value=float(value) if isinstance(value, (int, float)) else None,
                    formula="",
                    source_page=ev.get("page_number"),
                    source_text=ev.get("source_text", ""),
                    category=category,
                ))

        return cls(
            company_code=result.get("company_code", ""),
            company_name=result.get("company_name", ""),
            year=result.get("year", 0),
            industry=result.get("industry", ""),
            method="multi_agent",
            indicators=indicator_list,
            interpretation=result.get("interpretation", ""),
            metadata={
                k: result[k] for k in ("consistency_report", "extracted_values",
                                         "timestamp")
                if k in result
            },
        )
