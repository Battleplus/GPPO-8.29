"""LLM/RAG attack-region recommendation integrated into Perch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .geojson_utils import parse_llm_geojson, validate_geojson
from .knowledge_base import ExpertKnowledgeBase
from .llm_provider import BaseLLMProvider, create_llm_provider


SYSTEM_PROMPT_TEMPLATE = """你是一个直升机火力打击战术专家。根据输入的目标描述、地形信息和专家知识，
推荐最佳攻击阵位区域。区域将由后续 FREA 算法继续选择具体攻击阵位点。

## 专家知识

{expert_knowledge}

## 推理要求

1. 结合目标、地形遮蔽、武器射程、威胁和撤离路线选择区域。
2. 输出 1-3 个按优先级排列的候选多边形区域。
3. 区域应与武器射程相容，且有足够面积供后续阵位点优化。
4. 每个区域必须给出 0-1 的 score 和简短 reasoning。
5. 态势输入中的实时武器包线和任务数据优先于专家知识中的典型值。

## 输出契约

只输出 JSON，不要输出 Markdown 或额外说明：
{{
  "type": "FeatureCollection",
  "features": [
    {{
      "type": "Feature",
      "geometry": {{
        "type": "Polygon",
        "coordinates": [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon1, lat1]]]
      }},
      "properties": {{
        "score": 0.85,
        "reasoning": "该区域利用反斜面遮蔽并满足武器射程"
      }}
    }}
  ]
}}

坐标必须使用 [经度, 纬度]，多边形必须闭合，经纬度保留 6 位小数。"""


@dataclass(frozen=True)
class RegionRecommendation:
    success: bool
    geojson: dict[str, Any] | None
    errors: list[str]
    raw_output: str
    knowledge_sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "geojson": self.geojson,
            "errors": list(self.errors),
            "raw_output": self.raw_output,
            "knowledge_sources": list(self.knowledge_sources),
        }


class AttackRegionRecommender:
    """Retrieve doctrine, call an LLM, and validate its region output."""

    def __init__(
        self,
        llm: BaseLLMProvider | None = None,
        knowledge_base: ExpertKnowledgeBase | None = None,
        rag_top_k: int | None = None,
    ) -> None:
        self._llm = llm
        self._knowledge_base = knowledge_base or ExpertKnowledgeBase()
        self._rag_top_k = rag_top_k or _env_int("PERCH_RAG_TOP_K", 3)

    def recommend(self, description: str) -> dict[str, Any]:
        documents, sources = self._knowledge_base.retrieve(
            description,
            top_k=self._rag_top_k,
        )
        expert_knowledge = "\n\n---\n\n".join(documents)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            expert_knowledge=expert_knowledge or "（无匹配专家知识）"
        )
        user_prompt = (
            "请根据以下态势理解文本推荐攻击阵位区域。后续算法会在区域内"
            f"选择具体阵位点：\n\n{description}"
        )
        llm = self._llm or create_llm_provider()
        raw_output = llm.generate(system_prompt, user_prompt)
        geojson = parse_llm_geojson(raw_output)
        if geojson is None:
            return RegionRecommendation(
                False,
                None,
                ["Unable to parse GeoJSON from the LLM response"],
                raw_output,
                sources,
            ).to_dict()
        errors = validate_geojson(geojson)
        return RegionRecommendation(
            not errors,
            geojson,
            errors,
            raw_output,
            sources,
        ).to_dict()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default
