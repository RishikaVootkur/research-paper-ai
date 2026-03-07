"""
Paper Recommender
-----------------
Recommends similar papers using embedding similarity.

Given a paper (by ID or text), finds the most similar papers
in the database using cosine similarity on their embeddings.

This demonstrates content-based filtering — one of the core
recommendation system approaches.
"""

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.vector_store import VectorStore


class PaperRecommender:
    """
    Content-based paper recommendation using embedding similarity.

    For each paper, we compute an "average embedding" from all its chunks.
    To find similar papers, we compute cosine similarity between
    paper embeddings.

    Usage:
        recommender = PaperRecommender()
        similar = recommender.recommend_by_id("2005.11401v4", top_k=5)
        similar = recommender.recommend_by_text("transformers for NLP", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
    ):
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

        # Build paper-level embeddings from chunk embeddings
        print("Building paper embeddings...")
        self.paper_embeddings, self.paper_info = self._build_paper_embeddings()
        print(f"  Built embeddings for {len(self.paper_embeddings)} papers")

    def _build_paper_embeddings(self) -> tuple[dict, dict]:
        """
        Compute average embedding per paper from chunk embeddings.

        Each paper has multiple chunks in ChromaDB. We average their
        embeddings to get a single "paper embedding" that represents
        the paper's overall content.
        """
        # Get all data from ChromaDB
        all_data = self.vector_store.collection.get(
            include=["embeddings", "metadatas"]
        )

        # Group embeddings by paper_id
        paper_chunks = {}
        paper_info = {}

        for i, metadata in enumerate(all_data["metadatas"]):
            pid = metadata["paper_id"]

            if pid not in paper_chunks:
                paper_chunks[pid] = []
                paper_info[pid] = {
                    "paper_id": pid,
                    "title": metadata["paper_title"],
                    "authors": metadata.get("authors", "Unknown"),
                }

            paper_chunks[pid].append(all_data["embeddings"][i])

        # Average embeddings per paper
        paper_embeddings = {}
        for pid, embeddings in paper_chunks.items():
            paper_embeddings[pid] = np.mean(embeddings, axis=0)

        return paper_embeddings, paper_info

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recommend_by_id(self, paper_id: str, top_k: int = 5) -> list[dict]:
        """
        Find papers similar to a given paper.

        Args:
            paper_id: ArXiv paper ID
            top_k: Number of recommendations

        Returns:
            List of {"paper_id", "title", "authors", "similarity"} dicts
        """
        if paper_id not in self.paper_embeddings:
            print(f"Paper {paper_id} not found in database.")
            return []

        query_embedding = self.paper_embeddings[paper_id]

        # Compute similarity to all other papers
        similarities = []
        for pid, emb in self.paper_embeddings.items():
            if pid == paper_id:
                continue
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((pid, sim))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for pid, sim in similarities[:top_k]:
            info = self.paper_info[pid]
            results.append({
                "paper_id": pid,
                "title": info["title"],
                "authors": info["authors"],
                "similarity": round(float(sim), 4),
            })

        return results

    def recommend_by_text(self, text: str, top_k: int = 5) -> list[dict]:
        """
        Find papers similar to a given text description.

        Args:
            text: Description or abstract of a topic
            top_k: Number of recommendations

        Returns:
            List of similar papers
        """
        # Embed the query text
        query_embedding = self.vector_store.embedding_model.encode(text)

        # Compute similarity to all papers
        similarities = []
        for pid, emb in self.paper_embeddings.items():
            sim = self._cosine_similarity(query_embedding, emb)
            similarities.append((pid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for pid, sim in similarities[:top_k]:
            info = self.paper_info[pid]
            results.append({
                "paper_id": pid,
                "title": info["title"],
                "authors": info["authors"],
                "similarity": round(float(sim), 4),
            })

        return results


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    recommender = PaperRecommender()

    # Test 1: Papers similar to the LoRA paper
    print("\n" + "=" * 60)
    print("Papers similar to: LoRA")
    print("=" * 60)
    results = recommender.recommend_by_id("2106.09685v2", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.4f}] {r['title'][:60]}...")

    # Test 2: Papers similar to Attention Is All You Need
    print("\n" + "=" * 60)
    print("Papers similar to: Attention Is All You Need")
    print("=" * 60)
    results = recommender.recommend_by_id("1706.03762v7", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.4f}] {r['title'][:60]}...")

    # Test 3: Recommend by text description
    print("\n" + "=" * 60)
    print("Papers about: 'evaluation metrics for retrieval augmented generation'")
    print("=" * 60)
    results = recommender.recommend_by_text(
        "evaluation metrics for retrieval augmented generation systems",
        top_k=5,
    )
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.4f}] {r['title'][:60]}...")

    # Test 4: Recommend by text — different topic
    print("\n" + "=" * 60)
    print("Papers about: 'efficient training of neural networks with limited GPU'")
    print("=" * 60)
    results = recommender.recommend_by_text(
        "efficient training of neural networks with limited GPU memory",
        top_k=5,
    )
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['similarity']:.4f}] {r['title'][:60]}...")

    print("\n✅ Recommendation system working!")