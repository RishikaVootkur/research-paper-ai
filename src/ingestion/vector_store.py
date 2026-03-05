"""
Vector Store
-------------
Embeds paper chunks and stores them in ChromaDB for similarity search.
This is the bridge between raw text and intelligent retrieval.
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dataclasses import asdict
from tqdm import tqdm

# We need PaperChunk type — import from our own module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ingestion.pdf_processor import PaperChunk


class VectorStore:
    """
    Manages embedding and storage of paper chunks in ChromaDB.

    This is the core of our retrieval system. Every chunk gets:
    1. Embedded into a 384-dimensional vector
    2. Stored in ChromaDB with its metadata
    3. Made searchable by semantic similarity

    Usage:
        store = VectorStore()
        store.add_chunks(chunks)
        results = store.search("What is retrieval augmented generation?", top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            collection_name: Name of the ChromaDB collection (like a table name).
            persist_dir: Where ChromaDB saves data on disk.
            embedding_model: Which sentence-transformer model to use.
                            "all-MiniLM-L6-v2" is small, fast, and good enough.
                            We'll swap this for our fine-tuned model in Week 4.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Initialize the embedding model
        # This loads the model into memory (~80MB). It stays loaded
        # for the lifetime of this object, so subsequent calls are fast.
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"  Model loaded. Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB with persistent storage
        # persist_directory means data survives between runs —
        # you embed once, search many times
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Get or create the collection
        # A collection is like a table — it holds all our embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Research paper chunks"},
        )

        print(f"  ChromaDB collection '{collection_name}': {self.collection.count()} existing documents")

    def add_chunks(self, chunks: list[PaperChunk], batch_size: int = 50):
        """
        Embed and store chunks in ChromaDB.

        Args:
            chunks: List of PaperChunk objects from our PDF processor
            batch_size: How many chunks to process at once.
                       Batching is more efficient than one-at-a-time.
        """
        if not chunks:
            print("No chunks to add.")
            return

        # Filter out chunks that are already in the database
        # This prevents duplicates if you run the ingestion twice
        existing_ids = set(self.collection.get()["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            print(f"All {len(chunks)} chunks already exist in the database. Skipping.")
            return

        print(f"Embedding and storing {len(new_chunks)} new chunks (skipping {len(chunks) - len(new_chunks)} duplicates)...")

        # Process in batches
        for i in tqdm(range(0, len(new_chunks), batch_size), desc="Adding chunks"):
            batch = new_chunks[i : i + batch_size]

            # Extract text content for embedding
            texts = [chunk.content for chunk in batch]

            # Generate embeddings for the entire batch at once
            # This is much faster than embedding one at a time
            embeddings = self.embedding_model.encode(texts).tolist()

            # Prepare IDs (ChromaDB requires unique string IDs)
            ids = [chunk.chunk_id for chunk in batch]

            # Prepare metadata (ChromaDB stores this alongside embeddings)
            # This is what lets us filter and cite later
            metadatas = [
                {
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "char_count": chunk.char_count,
                    "authors": ", ".join(chunk.authors[:5]),  # ChromaDB metadata must be string/int/float
                }
                for chunk in batch
            ]

            # Add to ChromaDB — this stores the embedding + metadata + original text
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        print(f"Done. Total documents in collection: {self.collection.count()}")

    def search(self, query: str, top_k: int = 5, filter_paper_id: str = None) -> list[dict]:
        """
        Search for chunks similar to a query.

        This is the core retrieval function. It:
        1. Embeds your query into a vector
        2. Finds the top_k most similar chunks in ChromaDB
        3. Returns them with similarity scores

        Args:
            query: Natural language question or search terms
            top_k: How many results to return
            filter_paper_id: Optionally restrict search to one paper

        Returns:
            List of dicts, each containing:
                - content: The chunk text
                - metadata: Paper info (title, authors, page, etc.)
                - score: Similarity score (lower = more similar in ChromaDB's distance)
        """
        # Build optional filter
        where_filter = None
        if filter_paper_id:
            where_filter = {"paper_id": filter_paper_id}

        # Embed the query
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),  # Can't return more than we have
            where=where_filter,
        )

        # Format results into a clean list
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],  # Lower = more similar
            })

        return formatted

    def get_paper_ids(self) -> list[str]:
        """Get a list of all unique paper IDs in the database."""
        all_metadata = self.collection.get()["metadatas"]
        paper_ids = list(set(m["paper_id"] for m in all_metadata))
        return sorted(paper_ids)

    def get_collection_stats(self) -> dict:
        """Get statistics about the vector store."""
        total = self.collection.count()
        if total == 0:
            return {"total_chunks": 0, "papers": 0}

        all_metadata = self.collection.get()["metadatas"]
        paper_ids = set(m["paper_id"] for m in all_metadata)

        return {
            "total_chunks": total,
            "papers": len(paper_ids),
            "paper_ids": sorted(paper_ids),
        }

    def delete_paper(self, paper_id: str):
        """Remove all chunks for a specific paper from the database."""
        # Get IDs of chunks belonging to this paper
        all_data = self.collection.get(where={"paper_id": paper_id})
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
            print(f"Deleted {len(all_data['ids'])} chunks for paper {paper_id}")
        else:
            print(f"No chunks found for paper {paper_id}")

    def reset(self):
        """Delete the entire collection and start fresh."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Research paper chunks"},
        )
        print(f"Collection '{self.collection_name}' reset. Now empty.")


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    from arxiv_fetcher import ArxivFetcher
    from pdf_processor import PDFProcessor

    # --- Step 1: Fetch and process a paper ---
    print("=" * 60)
    print("STEP 1: Fetch and process papers")
    print("=" * 60)

    fetcher = ArxivFetcher()
    processor = PDFProcessor()

    # Fetch 3 landmark papers
    papers = fetcher.search_by_ids([
        "2005.11401",  # RAG paper
        "1706.03762",  # Attention Is All You Need
    ])

    downloaded = fetcher.download_papers(papers)

    all_chunks = []
    for paper in papers:
        pdf_path = downloaded.get(paper.paper_id)
        if pdf_path:
            chunks = processor.process_pdf(
                pdf_path=pdf_path,
                paper_id=paper.paper_id,
                paper_title=paper.title,
                authors=paper.authors,
            )
            all_chunks.extend(chunks)

    # --- Step 2: Store in vector database ---
    print("\n" + "=" * 60)
    print("STEP 2: Store chunks in vector database")
    print("=" * 60)

    store = VectorStore()
    store.add_chunks(all_chunks)

    # --- Step 3: Search ---
    print("\n" + "=" * 60)
    print("STEP 3: Test searches")
    print("=" * 60)

    queries = [
        "What is retrieval augmented generation?",
        "How does the attention mechanism work in transformers?",
        "What datasets were used for evaluation?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        results = store.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            meta = result["metadata"]
            print(f"\n  Result {i} (distance: {result['distance']:.4f}):")
            print(f"  Paper: {meta['paper_title'][:60]}...")
            print(f"  Page:  {meta['page_number']}")
            print(f"  Text:  {result['content'][:150]}...")

    # --- Step 4: Stats ---
    print("\n" + "=" * 60)
    print("STEP 4: Collection stats")
    print("=" * 60)
    stats = store.get_collection_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total papers: {stats['papers']}")
    print(f"Paper IDs:    {stats['paper_ids']}")

    # --- Step 5: Test duplicate prevention ---
    print("\n" + "=" * 60)
    print("STEP 5: Test duplicate prevention")
    print("=" * 60)
    store.add_chunks(all_chunks)  # Should skip all since they already exist

    print("\n✅ Vector store complete!")