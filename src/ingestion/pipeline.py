"""
Ingestion Pipeline
------------------
End-to-end pipeline: Search → Download → Extract → Chunk → Embed → Store

This ties together all three modules (ArxivFetcher, PDFProcessor, VectorStore)
into a single, easy-to-use pipeline.

Usage:
    # From command line:
    python src/ingestion/pipeline.py --query "retrieval augmented generation" --max-papers 5

    # From Python:
    from src.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline.ingest_by_query("attention mechanism transformers", max_papers=5)
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.arxiv_fetcher import ArxivFetcher, PaperMetadata
from src.ingestion.pdf_processor import PDFProcessor, PaperChunk
from src.ingestion.vector_store import VectorStore


class IngestionPipeline:
    """
    Orchestrates the full ingestion process.

    Think of this as the "manager" — it doesn't do the work itself,
    it coordinates the fetcher, processor, and vector store to work
    together smoothly.
    """

    def __init__(
        self,
        download_dir: str = "data/raw_papers",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
    ):
        self.fetcher = ArxivFetcher(download_dir=download_dir)
        self.processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        self.download_dir = download_dir

        # Keep a log of what we've ingested
        self.log_path = "data/processed/ingestion_log.json"
        self.log = self._load_log()

    def _load_log(self) -> dict:
        """Load the ingestion log (tracks what papers we've already processed)."""
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                return json.load(f)
        return {"ingested_papers": {}, "queries": []}

    def _save_log(self):
        """Save the ingestion log."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

    def _is_already_ingested(self, paper_id: str) -> bool:
        """Check if a paper has already been processed."""
        return paper_id in self.log["ingested_papers"]

    def ingest_papers(self, papers: list[PaperMetadata]) -> dict:
        """
        Process a list of papers through the full pipeline.

        This is the core method. It takes papers (from any source)
        and pushes them through: download → extract → chunk → embed → store.

        Args:
            papers: List of PaperMetadata objects

        Returns:
            Summary dict with counts of processed/skipped/failed papers
        """
        summary = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        for i, paper in enumerate(papers, 1):
            print(f"\n[{i}/{len(papers)}] {paper.title[:70]}...")

            # Skip if already ingested
            if self._is_already_ingested(paper.paper_id):
                print("  ⏭️  Already ingested — skipping")
                summary["skipped"] += 1
                continue

            try:
                # Step 1: Download PDF
                pdf_path = self.fetcher.download_pdf(paper)
                if not pdf_path:
                    print("  ❌ Download failed — skipping")
                    summary["failed"] += 1
                    continue

                # Step 2: Extract text and create chunks
                chunks = self.processor.process_pdf(
                    pdf_path=pdf_path,
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    authors=paper.authors,
                )

                if not chunks:
                    print("  ❌ No chunks created — skipping")
                    summary["failed"] += 1
                    continue

                # Step 3: Embed and store in vector database
                self.vector_store.add_chunks(chunks)

                # Step 4: Log it
                self.log["ingested_papers"][paper.paper_id] = {
                    "title": paper.title,
                    "authors": paper.authors[:5],
                    "chunks": len(chunks),
                    "ingested_at": datetime.now().isoformat(),
                }
                self._save_log()

                # Step 5: Clean up PDF to save disk space
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print(f"  🗑️  Cleaned up PDF")

                summary["processed"] += 1
                summary["total_chunks"] += len(chunks)
                print(f"  ✅ Done — {len(chunks)} chunks added")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                summary["failed"] += 1

        return summary

    def ingest_by_query(self, query: str, max_papers: int = 10) -> dict:
        """
        Search ArXiv and ingest the results.

        Args:
            query: Search terms
            max_papers: How many papers to fetch

        Returns:
            Summary dict
        """
        print(f"\n{'='*60}")
        print(f"Ingesting papers for: '{query}'")
        print(f"{'='*60}")

        # Fetch papers
        papers = self.fetcher.search(query, max_results=max_papers)

        if not papers:
            print("No papers found.")
            return {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        # Log the query
        self.log["queries"].append({
            "query": query,
            "max_papers": max_papers,
            "results_found": len(papers),
            "timestamp": datetime.now().isoformat(),
        })
        self._save_log()

        # Save metadata
        self.fetcher.save_metadata(papers)

        # Run the pipeline
        summary = self.ingest_papers(papers)

        self._print_summary(summary)
        return summary

    def ingest_by_ids(self, paper_ids: list[str]) -> dict:
        """
        Ingest specific papers by ArXiv ID.

        Args:
            paper_ids: List of ArXiv IDs like ["2005.11401", "1706.03762"]

        Returns:
            Summary dict
        """
        print(f"\n{'='*60}")
        print(f"Ingesting {len(paper_ids)} specific papers")
        print(f"{'='*60}")

        papers = self.fetcher.search_by_ids(paper_ids)
        summary = self.ingest_papers(papers)

        self._print_summary(summary)
        return summary

    def ingest_by_category(self, category: str, max_papers: int = 10) -> dict:
        """
        Ingest latest papers from an ArXiv category.

        Args:
            category: ArXiv category like "cs.CL", "cs.AI", "cs.LG"
            max_papers: How many papers to fetch

        Returns:
            Summary dict
        """
        print(f"\n{'='*60}")
        print(f"Ingesting latest papers from category: {category}")
        print(f"{'='*60}")

        papers = self.fetcher.search_by_category(category, max_results=max_papers)
        summary = self.ingest_papers(papers)

        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: dict):
        """Print a nice summary of the ingestion run."""
        print(f"\n{'='*60}")
        print("INGESTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Processed:    {summary['processed']} papers")
        print(f"  Skipped:      {summary['skipped']} (already ingested)")
        print(f"  Failed:       {summary['failed']}")
        print(f"  Total chunks: {summary['total_chunks']}")

        stats = self.vector_store.get_collection_stats()
        print(f"\n  Database now contains:")
        print(f"    {stats['total_chunks']} chunks from {stats['papers']} papers")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the ingested papers.
        Convenience method so you can search directly from the pipeline.
        """
        return self.vector_store.search(query, top_k=top_k)

    def show_status(self):
        """Show what's currently in the database."""
        stats = self.vector_store.get_collection_stats()
        print(f"\n{'='*60}")
        print("DATABASE STATUS")
        print(f"{'='*60}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total papers: {stats['papers']}")

        if stats.get("paper_ids"):
            print(f"\nIngested papers:")
            for pid in stats["paper_ids"]:
                info = self.log["ingested_papers"].get(pid, {})
                title = info.get("title", "Unknown")
                chunks = info.get("chunks", "?")
                print(f"  [{pid}] {title[:60]}... ({chunks} chunks)")


# ============================================================
# Command-line interface
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest research papers into the vector database"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Search query for ArXiv (e.g., 'retrieval augmented generation')"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Specific ArXiv paper IDs to ingest (e.g., 2005.11401 1706.03762)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        help="ArXiv category to fetch from (e.g., cs.CL, cs.AI, cs.LG)"
    )
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=5,
        help="Maximum number of papers to fetch (default: 5)"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        help="Search the existing database instead of ingesting"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current database status"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the entire database (WARNING: deletes everything)"
    )

    args = parser.parse_args()
    pipeline = IngestionPipeline()

    if args.reset:
        confirm = input("Are you sure you want to delete ALL data? (yes/no): ")
        if confirm.lower() == "yes":
            pipeline.vector_store.reset()
            print("Database reset.")
        else:
            print("Cancelled.")

    elif args.status:
        pipeline.show_status()

    elif args.search:
        results = pipeline.search(args.search, top_k=5)
        print(f"\n🔍 Results for: '{args.search}'")
        print("-" * 50)
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n  Result {i} (distance: {r['distance']:.4f})")
            print(f"  Paper: {meta['paper_title'][:60]}")
            print(f"  Page:  {meta['page_number']}")
            print(f"  Text:  {r['content'][:200]}...")

    elif args.ids:
        pipeline.ingest_by_ids(args.ids)

    elif args.query:
        pipeline.ingest_by_query(args.query, max_papers=args.max_papers)

    elif args.category:
        pipeline.ingest_by_category(args.category, max_papers=args.max_papers)

    else:
        parser.print_help()