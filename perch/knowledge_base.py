"""Lightweight local retrieval for helicopter attack-position doctrine."""

from __future__ import annotations

import re
from pathlib import Path


class ExpertKnowledgeBase:
    """Retrieve relevant Markdown doctrine without a persistent vector DB."""

    def __init__(self, docs_dir: str | Path | None = None) -> None:
        self.docs_dir = (
            Path(docs_dir)
            if docs_dir is not None
            else Path(__file__).with_name("expert_docs")
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[list[str], list[str]]:
        documents: list[tuple[float, str, str]] = []
        query_terms = _terms(query)
        for path in sorted(self.docs_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            doc_terms = _terms(content)
            overlap = len(query_terms & doc_terms)
            filename_bonus = sum(
                1 for token in _terms(path.stem) if token in query_terms
            )
            score = overlap + filename_bonus * 2
            if score > 0:
                documents.append((score, path.name, content))

        documents.sort(key=lambda item: (-item[0], item[1]))
        chosen = documents[: max(1, int(top_k))]
        return (
            [content for _, _, content in chosen],
            [name for _, name, _ in chosen],
        )


def _terms(text: str) -> set[str]:
    normalized = str(text).lower()
    latin = set(re.findall(r"[a-z0-9][a-z0-9_-]+", normalized))
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese = {
        run[index:index + 2]
        for run in chinese_runs
        for index in range(max(0, len(run) - 1))
    }
    return latin | chinese
