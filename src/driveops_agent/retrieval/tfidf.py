import re
from pathlib import Path


class OfflineRetriever:
    """Small deterministic TF-IDF-like lexical retriever; documents remain untrusted data."""

    def __init__(self, root: Path):
        self.root = root
        self.chunks = self._chunks()

    def _chunks(self):
        result = []
        for p in [self.root / "validation_notes.md"]:
            for i, chunk in enumerate(p.read_text(encoding="utf8").split("\n## ")):
                result.append((p.name, f"{p.stem}-{i}", chunk))
        return result

    def search(self, query: str, limit: int = 3):
        words = set(re.findall(r"[a-z0-9-]+", query.lower()))
        scored = []
        for source, chunk_id, content in self.chunks:
            score = len(words & set(re.findall(r"[a-z0-9-]+", content.lower()))) / max(
                len(words), 1
            )
            if score:
                scored.append(
                    {
                        "source_id": source,
                        "chunk_id": chunk_id,
                        "score": round(score, 3),
                        "content": content,
                    }
                )
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
