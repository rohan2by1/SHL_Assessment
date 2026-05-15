import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.embed_model = SentenceTransformer(embed_model)
        self.reranker = CrossEncoder(reranker_model)
        self.index = None
        self.catalog = []
        self.bm25 = None
        self.corpus = []

    def build_index(self, assessments: List[Dict]):
        self.catalog = assessments
        texts = [a["embedding_text"] for a in assessments]
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embeddings.astype('float32'))
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = tokenized_corpus

    def _vector_search(self, query: str, k: int = 40) -> List[Dict]:
        query_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.catalog):
                results.append((self.catalog[idx], float(dist)))
        return results

    def _bm25_search(self, query: str, k: int = 40) -> List[Dict]:
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(self.catalog[i], scores[i]) for i in top_indices]
        return results

    def _fuse_results(self, vec_res: List, bm25_res: List, k: int = 20) -> List[Dict]:
        fused = {}
        for rank, (item, score) in enumerate(vec_res):
            url = item["url"]
            fused[url] = fused.get(url, 0) + 1 / (rank + 60)
        for rank, (item, score) in enumerate(bm25_res):
            url = item["url"]
            fused[url] = fused.get(url, 0) + 1 / (rank + 60)
        sorted_urls = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        results = []
        for url, _ in sorted_urls[:k]:
            item = next((a for a in self.catalog if a["url"] == url), None)
            if item:
                results.append(item)
        return results

    def search_with_rerank(self, queries: List[str], filters: Optional[Dict] = None,
                           k: int = 10, fuse_k: int = 30) -> List[Dict]:
        """
        Multi-query (dual-pass) retrieval with reranking.
        queries: list of query strings (e.g., technical + soft skills).
        """
        # Collect candidates from each query
        candidates = []
        seen_urls = set()
        for q in queries:
            vec_res = self._vector_search(q, fuse_k)
            bm25_res = self._bm25_search(q, fuse_k)
            fused = self._fuse_results(vec_res, bm25_res, fuse_k)
            for item in fused:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    candidates.append(item)
        # Apply filters
        if filters:
            candidates = [c for c in candidates if self._passes_filters(c, filters)]
        if not candidates:
            return []
        # Rerank
        # We'll use a combined query: join all queries with " ; "
        combined_query = " ; ".join(queries)
        pairs = [(combined_query, c["embedding_text"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_candidates = [c for c, s in ranked[:k]]
        return top_candidates

    def _passes_filters(self, item: Dict, filters: Dict) -> bool:
        if "test_type_codes" in filters and filters["test_type_codes"]:
            if not any(tc in item.get("test_type_codes", []) for tc in filters["test_type_codes"]):
                return False
        if "job_levels" in filters and filters["job_levels"]:
            req = [l.lower() for l in filters["job_levels"]]
            item_levels = [l.lower() for l in item.get("job_levels", [])]
            if not any(any(rl in il for il in item_levels) for rl in req):
                return False
        return True

    def get_by_name(self, name: str) -> Optional[Dict]:
        name_lower = name.lower().strip()
        for a in self.catalog:
            if a["title"].lower() == name_lower:
                return a
        return None

    def get_by_url(self, url: str) -> Optional[Dict]:
        for a in self.catalog:
            if a["url"] == url:
                return a
        return None