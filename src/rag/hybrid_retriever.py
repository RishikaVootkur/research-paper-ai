"""
Hybrid Retriever
----------------
Combines vector search (semantic) with BM25 (keyword) search
using Reciprocal Rank Fusion for better retrieval quality.

Why hybrid?
- Vector search understands meaning but misses exact keywords
- BM25 finds exact keyword matches but doesn't understand meaning
- Together they cover each other's blind spots
"""

import os
import sys
import math
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """
    A single retrieved chunk with all its info.
    Using a dataclass instead of raw dicts makes the code cleaner
    and easier to work with downstream.
    """
    chunk_id: str
    content: str
    paper_id: str
    paper_title: str
    authors: str
    page_number: int
    score: float           # Combined retrieval score (higher = better)
    retrieval_method: str  # "hybrid", "vector_only", or "bm25_only"


class HybridRetriever:
    """
    Combines ChromaDB vector search with BM25 keyword search.

    The key idea: run both searches, then merge results using
    Reciprocal Rank Fusion (RRF). Chunks that rank highly in
    BOTH methods get boosted to the top.

    Usage:
        retriever = HybridRetriever()
        results = retriever.search("How does LoRA work?", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Args:
            collection_name: ChromaDB collection name
            persist_dir: ChromaDB storage path
            vector_weight: Weight for vector search results (0.0 to 1.0)
            bm25_weight: Weight for BM25 results (0.0 to 1.0)
                        These two weights control the balance.
                        0.5/0.5 = equal weight (good default)
                        0.7/0.3 = favor semantic meaning
                        0.3/0.7 = favor keyword matching
        """
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

        # Load all documents from ChromaDB for BM25 indexing
        print("Building BM25 index...")
        self._build_bm25_index()
        print(f"  BM25 index built with {len(self.corpus)} documents")

    def _build_bm25_index(self):
        """
        Build the BM25 index from all documents in ChromaDB.

        BM25 needs ALL documents loaded in memory to calculate
        term frequencies. This is fine for our scale (a few thousand
        chunks) but wouldn't work for millions of documents.
        """
        # Get all documents from ChromaDB
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        self.corpus_ids = all_data["ids"]
        self.corpus = all_data["documents"]
        self.corpus_metadata = all_data["metadatas"]

        # Tokenize documents for BM25
        # Simple whitespace + lowercase tokenization
        # More sophisticated tokenization could improve results,
        # but this works well enough for academic text
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        # Build the BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Run vector (semantic) search via ChromaDB."""
        results = self.vector_store.search(query, top_k=top_k)

        # Normalize: convert distance to similarity score (0 to 1, higher = better)
        # ChromaDB returns distance where lower = more similar
        # We convert to: score = 1 / (1 + distance)
        for r in results:
            r["score"] = 1.0 / (1.0 + r["distance"])

        return results

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Run BM25 (keyword) search."""
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (highest first)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include if there's some match
                results.append({
                    "chunk_id": self.corpus_ids[idx],
                    "content": self.corpus[idx],
                    "metadata": self.corpus_metadata[idx],
                    "score": bm25_scores[idx],
                })

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for each list

        The idea: a chunk ranked #1 in a list gets score 1/(60+1) = 0.0164
        A chunk ranked #5 gets 1/(60+5) = 0.0154
        A chunk that appears in BOTH lists gets both scores added.

        This naturally boosts chunks that appear in both searches.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (60 is standard, higher = less emphasis on top ranks)

        Returns:
            Merged and re-ranked results
        """
        chunk_scores = {}  # chunk_id -> {"score": float, "data": dict, "methods": set}

        # Score vector results
        for rank, result in enumerate(vector_results, 1):
            cid = result["chunk_id"]
            rrf_score = self.vector_weight * (1.0 / (k + rank))

            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "score": 0,
                    "data": result,
                    "methods": set(),
                }
            chunk_scores[cid]["score"] += rrf_score
            chunk_scores[cid]["methods"].add("vector")

        # Score BM25 results
        for rank, result in enumerate(bm25_results, 1):
            cid = result["chunk_id"]
            rrf_score = self.bm25_weight * (1.0 / (k + rank))

            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "score": 0,
                    "data": result,
                    "methods": set(),
                }
            chunk_scores[cid]["score"] += rrf_score
            chunk_scores[cid]["methods"].add("bm25")

        # Sort by combined score (highest first)
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return sorted_chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
    ) -> list[RetrievedChunk]:
        """
        Perform hybrid search: vector + BM25 with RRF fusion.

        Args:
            query: Search query
            top_k: Number of final results to return
            fetch_k: Number of results to fetch from EACH method
                    before fusion. We fetch more than top_k because
                    fusion re-ranks and some results overlap.

        Returns:
            List of RetrievedChunk objects, sorted by relevance
        """
        # Run both searches (fetch more than needed for better fusion)
        vector_results = self._vector_search(query, top_k=fetch_k)
        bm25_results = self._bm25_search(query, top_k=fetch_k)

        # Fuse results
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Convert to RetrievedChunk objects and take top_k
        chunks = []
        for item in fused[:top_k]:
            data = item["data"]
            meta = data["metadata"]
            methods = item["methods"]

            # Determine retrieval method label
            if "vector" in methods and "bm25" in methods:
                method = "hybrid"
            elif "vector" in methods:
                method = "vector_only"
            else:
                method = "bm25_only"

            chunks.append(RetrievedChunk(
                chunk_id=data["chunk_id"],
                content=data["content"],
                paper_id=meta["paper_id"],
                paper_title=meta["paper_title"],
                authors=meta.get("authors", "Unknown"),
                page_number=meta["page_number"],
                score=item["score"],
                retrieval_method=method,
            ))

        return chunks

    def compare_methods(self, query: str, top_k: int = 5):
        """
        Compare vector-only, BM25-only, and hybrid results side by side.
        Useful for understanding how hybrid improves retrieval.
        """
        print(f"\n{'='*70}")
        print(f"COMPARISON: '{query}'")
        print(f"{'='*70}")

        # Vector only
        vector_results = self._vector_search(query, top_k=top_k)
        print(f"\n📐 VECTOR SEARCH (semantic):")
        for i, r in enumerate(vector_results, 1):
            meta = r["metadata"]
            print(f"  {i}. [{meta['paper_id']}] p.{meta['page_number']} "
                  f"(dist: {r['distance']:.4f}) — {meta['paper_title'][:50]}...")

        # BM25 only
        bm25_results = self._bm25_search(query, top_k=top_k)
        print(f"\n📝 BM25 SEARCH (keyword):")
        for i, r in enumerate(bm25_results, 1):
            meta = r["metadata"]
            print(f"  {i}. [{meta['paper_id']}] p.{meta['page_number']} "
                  f"(score: {r['score']:.4f}) — {meta['paper_title'][:50]}...")

        # Hybrid
        hybrid_results = self.search(query, top_k=top_k)
        print(f"\n🔀 HYBRID SEARCH (fused):")
        for i, r in enumerate(hybrid_results, 1):
            print(f"  {i}. [{r.paper_id}] p.{r.page_number} "
                  f"(score: {r.score:.6f}, method: {r.retrieval_method}) "
                  f"— {r.paper_title[:50]}...")

        # Show overlap stats
        vector_ids = {r["chunk_id"] for r in vector_results}
        bm25_ids = {r["chunk_id"] for r in bm25_results}
        overlap = vector_ids & bm25_ids
        print(f"\n📊 Overlap: {len(overlap)}/{top_k} chunks appear in both searches")


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    retriever = HybridRetriever()

    # Compare methods on different query types
    queries = [
        "LoRA low-rank adaptation fine-tuning",     # Has specific keywords
        "How do language models handle long contexts?",  # More semantic/conceptual
        "RAGAS evaluation metrics for RAG",          # Acronym-heavy
    ]

    for query in queries:
        retriever.compare_methods(query, top_k=5)