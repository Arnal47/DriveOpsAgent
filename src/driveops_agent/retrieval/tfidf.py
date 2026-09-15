import math
import re
from collections import Counter
from pathlib import Path


class OfflineRetriever:
    def __init__(self, root: Path, backend="tfidf"):
        if backend not in {"tfidf", "bm25", "hybrid"}:
            raise ValueError("unknown retrieval backend")
        self.root = root
        self.backend = backend
        self.chunks = self._chunks()

    def _chunks(self):
        out = []
        for p in sorted(self.root.glob("*.md")):
            for i, c in enumerate(p.read_text(encoding="utf8").split("\n## ")):
                out.append((p.name, f"{p.stem}-{i}", c))
        return out

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9-]+", text.lower())

    def search(self, query, limit=3):
        q = self._tokens(query)
        docs = [self._tokens(c) for _, _, c in self.chunks]
        avg = sum(map(len, docs)) / max(1, len(docs))
        scored = []
        for idx, (source, cid, content) in enumerate(self.chunks):
            counts = Counter(docs[idx])
            lexical = len(set(q) & set(counts)) / max(1, len(set(q)))
            bm25 = 0.0
            for word in set(q):
                df = sum(word in d for d in docs)
                idf = math.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
                tf = counts[word]
                bm25 += (
                    idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * len(docs[idx]) / max(1, avg)))
                    if tf
                    else 0
                )
            score = (
                lexical
                if self.backend == "tfidf"
                else bm25
                if self.backend == "bm25"
                else 0.5 * lexical + 0.5 * bm25
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
