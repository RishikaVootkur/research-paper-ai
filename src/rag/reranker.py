"""
Cross-Encoder Re-ranker
-----------------------
Takes the top-N results from hybrid search and re-ranks them
using a cross-encoder model that scores query-document pairs together.

This is the key quality improvement: bi-encoders (vector/BM25) encode
query and documents separately. Cross-encoders read them TOGETHER,
giving much more accurate relevance scores.

The tradeoff: cross-encoders are slower (can't pre-compute),
so we only run them on the top candidates from hybrid search.
"""

import os
import sys
from sentence_transformers import CrossEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.hybrid_retriever import HybridRetriever, RetrievedChunk


class Reranker:
    """
    Re-ranks retrieved chunks using a cross-encoder model.

    Cross-encoders take a (query, document) pair as input and output
    a relevance score. Unlike bi-encoders, they see both texts together,
    which allows them to capture fine-grained relevance signals.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace cross-encoder model.
                       "ms-marco-MiniLM-L-6-v2" is small (~80MB), fast,
                       and trained on MS MARCO (a search relevance dataset).
                       It's the standard choice for re-ranking.
        """
        print(f"Loading re-ranker model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("  Re-ranker loaded.")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Re-rank chunks by cross-encoder relevance score.

        Args:
            query: The user's question
            chunks: Pre-filtered chunks from hybrid search
            top_k: How many to keep after re-ranking

        Returns:
            Top-k chunks, re-ordered by cross-encoder score
        """
        if not chunks:
            return []

        # Create (query, document) pairs for the cross-encoder
        pairs = [[query, chunk.content] for chunk in chunks]

        # Score all pairs — this is where the cross-encoder shines.
        # It reads query + document together and outputs a relevance score.
        scores = self.model.predict(pairs)

        # Attach scores to chunks and sort by score (highest first)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Update chunk scores and return top_k
        reranked = []
        for chunk, score in scored_chunks[:top_k]:
            # Create a new chunk with the cross-encoder score
            reranked.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                paper_id=chunk.paper_id,
                paper_title=chunk.paper_title,
                authors=chunk.authors,
                page_number=chunk.page_number,
                score=float(score),  # Cross-encoder score
                retrieval_method=chunk.retrieval_method,
            ))

        return reranked


class RerankedRetriever:
    """
    Full retrieval pipeline: Hybrid Search → Cross-Encoder Re-rank.

    This is the final retriever that our RAG chain will use.
    It combines the speed of hybrid search with the precision
    of cross-encoder re-ranking.

    Pipeline:
        Query → Hybrid Search (fetch 20) → Re-rank → Return top 5
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.hybrid_retriever = HybridRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
        self.reranker = Reranker(model_name=reranker_model)

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
    ) -> list[RetrievedChunk]:
        """
        Full pipeline: hybrid search then re-rank.

        Args:
            query: User's question
            top_k: Final number of chunks to return
            fetch_k: How many to fetch from hybrid search before re-ranking.
                    Higher = better quality but slower.
                    20 is the sweet spot for our scale.

        Returns:
            Top-k re-ranked chunks
        """
        # Step 1: Cast a wide net with hybrid search
        candidates = self.hybrid_retriever.search(query, top_k=fetch_k)

        # Step 2: Re-rank with cross-encoder for precision
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        return reranked

    def compare_with_without_reranking(self, query: str, top_k: int = 5):
        """
        Show the difference re-ranking makes.
        Compares hybrid-only vs hybrid+reranking results.
        """
        print(f"\n{'='*70}")
        print(f"QUERY: '{query}'")
        print(f"{'='*70}")

        # Without re-ranking (just hybrid)
        hybrid_only = self.hybrid_retriever.search(query, top_k=top_k)
        print(f"\n📋 HYBRID ONLY (no re-ranking):")
        for i, r in enumerate(hybrid_only, 1):
            print(f"  {i}. [{r.paper_id}] p.{r.page_number} "
                  f"(score: {r.score:.6f}) — {r.paper_title[:50]}...")
            print(f"     Preview: {r.content[:100]}...")

        # With re-ranking
        reranked = self.search(query, top_k=top_k)
        print(f"\n🎯 HYBRID + RE-RANKING:")
        for i, r in enumerate(reranked, 1):
            print(f"  {i}. [{r.paper_id}] p.{r.page_number} "
                  f"(score: {r.score:.4f}) — {r.paper_title[:50]}...")
            print(f"     Preview: {r.content[:100]}...")

        # Show how much the ranking changed
        hybrid_ids = [r.chunk_id for r in hybrid_only]
        reranked_ids = [r.chunk_id for r in reranked]

        moved = 0
        for i, cid in enumerate(reranked_ids):
            if cid in hybrid_ids:
                old_pos = hybrid_ids.index(cid)
                if old_pos != i:
                    moved += 1

        new_entries = len(set(reranked_ids) - set(hybrid_ids))
        print(f"\n📊 Re-ranking impact: {moved} chunks changed position, "
              f"{new_entries} new chunks entered top {top_k}")


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    retriever = RerankedRetriever()

    queries = [
        "How does LoRA reduce the number of trainable parameters?",
        "What metrics are used to evaluate RAG systems?",
        "Explain multi-head attention in transformers",
    ]

    for query in queries:
        retriever.compare_with_without_reranking(query, top_k=5)