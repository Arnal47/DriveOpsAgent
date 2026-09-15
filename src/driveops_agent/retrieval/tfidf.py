import re
from pathlib import Path


class OfflineRetriever:
    def __init__(self, root: Path):
        self.root = root
        self.chunks = self._chunks()

    def _chunks(self):
        out = []
        for p in sorted(self.root.glob("*.md")):
            for i, c in enumerate(p.read_text(encoding="utf8").split("\n## ")):
                out.append((p.name, f"{p.stem}-{i}", c))
        return out

    def search(self, query, limit=3):
        words = set(re.findall(r"[a-z0-9-]+", query.lower()))
        scored = []
        for source, cid, content in self.chunks:
            score = len(words & set(re.findall(r"[a-z0-9-]+", content.lower()))) / max(
                1, len(words)
            )
            if score:
                scored.append(
                    {
                        "source_id": source,
                        "chunk_id": cid,
                        "score": round(score, 3),
                        "content": content,
                    }
                )
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]
