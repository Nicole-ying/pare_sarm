"""
episodic_memory.py — Per-experiment searchable round-level memory.

Layer 2 of the three-layer memory system. Stores all round artifacts
(reward functions, eval histories, agent outputs) with structured
indexing for retrieval.

Interface:
- store_round(round_num, artifacts) → index and persist
- search(query, k=5) → list of relevant entries
- get_round_summary(round_num) → compact round summary

Search: keyword + TF-IDF-like weighting (extensible to embeddings).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional


class EpisodicMemory:
    """Per-experiment episodic memory with searchable round storage.

    Each round's artifacts are stored on disk and indexed in memory.
    Search uses embedding-based semantic search when sentence-transformers
    is available, falling back to TF-IDF keyword scoring.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index: list[dict] = []  # [{round_num, summary, reward_src, path}]
        self._doc_freq: Counter = Counter()  # term → doc count for IDF
        self._loaded = False
        self._embed_model = None
        self._embeddings: list = []  # cached embeddings for semantic search

    # ── Storage ─────────────────────────────────────────────────────────────

    def store_round(self, round_num: int, artifacts: dict) -> None:
        """Store a round's artifacts and update the search index.

        Args:
            round_num: Round number.
            artifacts: Dict with keys like 'reward_fn_source', 'eval_summary',
                       'perception_report', 'reflection', etc.
        """
        round_dir = self.memory_dir / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Persist artifacts
        for key, value in artifacts.items():
            if isinstance(value, str):
                (round_dir / f"{key}.txt").write_text(value, encoding="utf-8")
            elif isinstance(value, dict):
                (round_dir / f"{key}.json").write_text(
                    json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        # Build index entry
        entry = {
            "round_num": round_num,
            "summary": artifacts.get("summary", ""),
            "reward_src": artifacts.get("reward_fn_source", "")[:2000],
            "perception": artifacts.get("perception_report", "")[:2000],
            "reflection": artifacts.get("reflection", "")[:2000],
            "eval_summary": artifacts.get("eval_summary", ""),
            "path": str(round_dir),
        }
        self._index.append(entry)
        self._update_doc_freq(entry, add=True)
        self._save_index()

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search episodic memory using semantic embeddings (if available) or TF-IDF.

        Args:
            query: Natural language query.
            k: Maximum number of results.

        Returns:
            List of {round_num, score, snippet, reward_src, ...} dicts.
        """
        if not self._index:
            return []

        # Try semantic search first
        if self._ensure_embed_model():
            return self._semantic_search(query, k)

        # Fall back to TF-IDF keyword search
        return self._tfidf_search(query, k)

    def _ensure_embed_model(self) -> bool:
        """Lazy-load sentence-transformers for semantic search."""
        if self._embed_model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            # Build embeddings for existing index entries
            self._build_embeddings()
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _build_embeddings(self) -> None:
        """Build embedding vectors for all indexed rounds."""
        if not self._embed_model:
            return
        texts = []
        for entry in self._index:
            doc = f"{entry.get('summary', '')} {entry.get('reflection', '')} {entry.get('reward_src', '')[:500]}"
            texts.append(doc)
        if texts:
            self._embeddings = self._embed_model.encode(texts, convert_to_numpy=True)

    def _semantic_search(self, query: str, k: int) -> list[dict]:
        """Cosine-similarity semantic search over embeddings."""
        if not self._embed_model or len(self._embeddings) == 0:
            return self._tfidf_search(query, k)

        import numpy as np
        query_vec = self._embed_model.encode([query], convert_to_numpy=True)[0]
        scores = np.dot(self._embeddings, query_vec) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        ranked = np.argsort(scores)[::-1][:k]

        results = []
        for idx in ranked:
            score = float(scores[idx])
            if score > 0.1:
                entry = self._index[idx]
                results.append({
                    "round_num": entry["round_num"],
                    "score": round(score, 4),
                    "snippet": entry.get("summary", "")[:300],
                    "reward_src": entry.get("reward_src", "")[:500],
                    "reflection": entry.get("reflection", "")[:300],
                })

        if not results:
            return self._tfidf_search(query, k)
        return results

    def _tfidf_search(self, query: str, k: int) -> list[dict]:
        """TF-IDF keyword search fallback."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return self._recent(k)

        n_docs = len(self._index)
        scores = []
        for entry in self._index:
            doc_text = f"{entry['summary']} {entry['reward_src']} {entry['reflection']}"
            doc_terms = self._tokenize(doc_text)
            score = self._tfidf_score(query_terms, doc_terms, n_docs)
            scores.append((score, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, entry in scores[:k]:
            if score > 0:
                results.append({
                    "round_num": entry["round_num"],
                    "score": round(score, 4),
                    "snippet": entry["summary"][:300],
                    "reward_src": entry["reward_src"][:500],
                    "reflection": entry["reflection"][:300],
                })

        if not results:
            return self._recent(k)
        return results

    def get_round_summary(self, round_num: int) -> Optional[dict]:
        """Get a compact summary of a specific round."""
        for entry in self._index:
            if entry["round_num"] == round_num:
                return {
                    "round_num": round_num,
                    "summary": entry["summary"],
                    "eval_summary": entry["eval_summary"],
                    "has_reflection": bool(entry.get("reflection")),
                }
        return None

    def get_recent_rounds(self, n: int = 3) -> list[dict]:
        """Get the n most recent rounds."""
        return self._recent(n)

    # ── Internal ────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric, min length 2."""
        return [t.lower() for t in re.findall(r'[a-zA-Z_]\w+', text.lower()) if len(t) >= 2]

    def _tfidf_score(self, query_terms: list[str], doc_terms: list[str], n_docs: int) -> float:
        """Compute TF-IDF-like score for a query against a document."""
        score = 0.0
        doc_counter = Counter(doc_terms)
        doc_len = max(len(doc_terms), 1)
        for term in query_terms:
            tf = doc_counter.get(term, 0) / doc_len
            df = self._doc_freq.get(term, 1)
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            score += tf * idf
        return score

    def _update_doc_freq(self, entry: dict, add: bool = True) -> None:
        """Update document frequency counts."""
        doc_text = f"{entry['summary']} {entry['reward_src']}"
        terms = set(self._tokenize(doc_text))
        for term in terms:
            if add:
                self._doc_freq[term] += 1
            else:
                self._doc_freq[term] = max(0, self._doc_freq[term] - 1)

    def _recent(self, k: int) -> list[dict]:
        """Return k most recent rounds."""
        recent = sorted(self._index, key=lambda e: e["round_num"], reverse=True)[:k]
        return [
            {
                "round_num": e["round_num"],
                "score": 0.0,
                "snippet": e["summary"][:300],
                "reward_src": e["reward_src"][:500],
                "reflection": e.get("reflection", "")[:300],
            }
            for e in recent
        ]

    def _save_index(self) -> None:
        """Persist the search index to disk (lightweight JSON)."""
        index_path = self.memory_dir / "episodic_index.json"
        # Strip large text fields from the persisted index
        slim = []
        for e in self._index:
            slim.append({
                "round_num": e["round_num"],
                "summary": e["summary"][:500],
                "reward_src": e["reward_src"][:500],
            })
        index_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index(self) -> None:
        """Load the search index from disk."""
        if self._loaded:
            return
        index_path = self.memory_dir / "episodic_index.json"
        if index_path.exists():
            try:
                slim = json.loads(index_path.read_text("utf-8"))
                self._index = slim
                for e in self._index:
                    self._update_doc_freq(e, add=True)
            except Exception:
                pass
        self._loaded = True
