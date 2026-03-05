"""
ArXiv Paper Fetcher
-------------------
Fetches research papers from ArXiv based on search queries.
Returns structured metadata for each paper.
"""

import arxiv
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PaperMetadata:
    """
    Represents one research paper's metadata.

    We use a dataclass here — think of it as a structured container.
    Instead of passing around messy dictionaries, we have a clean object
    with defined fields. This makes our code predictable and easier to debug.
    """
    paper_id: str           # ArXiv ID like "2312.10997"
    title: str              # Paper title
    abstract: str           # Paper abstract/summary
    authors: list[str]      # List of author names
    published: str          # Publication date as string
    updated: str            # Last updated date
    categories: list[str]   # ArXiv categories like ["cs.CL", "cs.AI"]
    primary_category: str   # Main category
    pdf_url: str            # Direct link to PDF
    arxiv_url: str          # Link to ArXiv page


class ArxivFetcher:
    """
    Handles searching and fetching papers from ArXiv.

    Usage:
        fetcher = ArxivFetcher()
        papers = fetcher.search("retrieval augmented generation", max_results=10)
    """

    def __init__(self, download_dir: str = "data/raw_papers"):
        """
        Args:
            download_dir: Where to save downloaded PDFs.
        """
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> list[PaperMetadata]:
        """
        Search ArXiv for papers matching a query.

        Args:
            query: Search terms (e.g., "LLM hallucination detection")
            max_results: How many papers to fetch (default 10)
            sort_by: How to sort — "relevance" or "date"

        Returns:
            List of PaperMetadata objects
        """
        # Map our simple sort names to ArXiv's sort criteria
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "date": arxiv.SortCriterion.SubmittedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        # Create the search — this doesn't actually make the API call yet
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        # Now fetch results. The arxiv library handles pagination and rate limiting.
        papers = []
        client = arxiv.Client()

        for result in client.results(search):
            paper = PaperMetadata(
                paper_id=result.entry_id.split("/")[-1],  # Extract ID from URL
                title=result.title.replace("\n", " "),     # Clean up newlines in titles
                abstract=result.summary.replace("\n", " "),
                authors=[author.name for author in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                updated=result.updated.strftime("%Y-%m-%d"),
                categories=[cat for cat in result.categories],
                primary_category=result.primary_category,
                pdf_url=result.pdf_url,
                arxiv_url=result.entry_id,
            )
            papers.append(paper)

        print(f"Found {len(papers)} papers for query: '{query}'")
        return papers

    def search_by_category(
        self,
        category: str,
        max_results: int = 10
    ) -> list[PaperMetadata]:
        """
        Search papers by ArXiv category.

        Common CS categories:
            cs.AI  - Artificial Intelligence
            cs.CL  - Computation and Language (NLP)
            cs.CV  - Computer Vision
            cs.LG  - Machine Learning
            cs.IR  - Information Retrieval

        Args:
            category: ArXiv category code
            max_results: How many papers to fetch
        """
        # ArXiv query syntax: cat:cs.CL means "category is cs.CL"
        query = f"cat:{category}"
        return self.search(query, max_results=max_results, sort_by="date")

    def search_by_ids(self, paper_ids: list[str]) -> list[PaperMetadata]:
        """
        Fetch specific papers by their ArXiv IDs.

        Args:
            paper_ids: List of ArXiv IDs like ["2312.10997", "2005.11401"]
        """
        search = arxiv.Search(id_list=paper_ids)
        client = arxiv.Client()

        papers = []
        for result in client.results(search):
            paper = PaperMetadata(
                paper_id=result.entry_id.split("/")[-1],
                title=result.title.replace("\n", " "),
                abstract=result.summary.replace("\n", " "),
                authors=[author.name for author in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                updated=result.updated.strftime("%Y-%m-%d"),
                categories=[cat for cat in result.categories],
                primary_category=result.primary_category,
                pdf_url=result.pdf_url,
                arxiv_url=result.entry_id,
            )
            papers.append(paper)

        print(f"Fetched {len(papers)} papers by ID")
        return papers

    def download_pdf(self, paper: PaperMetadata) -> Optional[str]:
        """
        Download a paper's PDF to the local directory.

        Args:
            paper: PaperMetadata object

        Returns:
            File path of the downloaded PDF, or None if download fails
        """
        try:
            # Create a clean filename from the paper ID
            filename = f"{paper.paper_id.replace('/', '_')}.pdf"
            filepath = os.path.join(self.download_dir, filename)

            # Skip if already downloaded
            if os.path.exists(filepath):
                print(f"  Already exists: {filename}")
                return filepath

            # Download using the arxiv library's built-in method
            # We need to get the actual arxiv.Result object for downloading
            search = arxiv.Search(id_list=[paper.paper_id])
            client = arxiv.Client()
            result = next(client.results(search))
            result.download_pdf(dirpath=self.download_dir, filename=filename)

            print(f"  Downloaded: {filename}")
            return filepath

        except Exception as e:
            print(f"  Failed to download {paper.paper_id}: {e}")
            return None

    def download_papers(self, papers: list[PaperMetadata]) -> dict[str, str]:
        """
        Download PDFs for a list of papers.

        Args:
            papers: List of PaperMetadata objects

        Returns:
            Dictionary mapping paper_id -> file_path for successful downloads
        """
        print(f"\nDownloading {len(papers)} papers...")
        downloaded = {}

        for i, paper in enumerate(papers, 1):
            print(f"[{i}/{len(papers)}] {paper.title[:60]}...")
            filepath = self.download_pdf(paper)
            if filepath:
                downloaded[paper.paper_id] = filepath

        print(f"\nSuccessfully downloaded {len(downloaded)}/{len(papers)} papers")
        return downloaded

    def save_metadata(
        self,
        papers: list[PaperMetadata],
        filepath: str = "data/processed/papers_metadata.json"
    ):
        """
        Save paper metadata to a JSON file for later use.

        Args:
            papers: List of PaperMetadata objects
            filepath: Where to save the JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert dataclasses to dictionaries for JSON serialization
        data = [asdict(paper) for paper in papers]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved metadata for {len(papers)} papers to {filepath}")

    @staticmethod
    def load_metadata(filepath: str = "data/processed/papers_metadata.json") -> list[PaperMetadata]:
        """
        Load previously saved paper metadata from JSON.

        Args:
            filepath: Path to the JSON file

        Returns:
            List of PaperMetadata objects
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        papers = [PaperMetadata(**item) for item in data]
        print(f"Loaded metadata for {len(papers)} papers from {filepath}")
        return papers


# ============================================================
# This block runs only when you execute this file directly
# (not when you import it from another file)
# ============================================================
if __name__ == "__main__":
    fetcher = ArxivFetcher()

    # --- Test 1: Search by keyword ---
    print("=" * 60)
    print("TEST 1: Search by keyword")
    print("=" * 60)
    papers = fetcher.search("retrieval augmented generation LLM", max_results=5)

    for i, paper in enumerate(papers, 1):
        print(f"\n--- Paper {i} ---")
        print(f"Title:    {paper.title}")
        print(f"ID:       {paper.paper_id}")
        print(f"Date:     {paper.published}")
        print(f"Category: {paper.primary_category}")
        print(f"Authors:  {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"Abstract: {paper.abstract[:200]}...")
        print(f"PDF:      {paper.pdf_url}")

    # --- Test 2: Fetch specific landmark papers ---
    print("\n" + "=" * 60)
    print("TEST 2: Fetch specific papers by ID")
    print("=" * 60)
    landmark_papers = fetcher.search_by_ids([
        "2005.11401",  # RAG original paper
        "1706.03762",  # Attention Is All You Need
    ])

    for paper in landmark_papers:
        print(f"\n  {paper.title}")
        print(f"  Published: {paper.published}")

    # --- Test 3: Download one paper ---
    print("\n" + "=" * 60)
    print("TEST 3: Download a PDF")
    print("=" * 60)
    if papers:
        filepath = fetcher.download_pdf(papers[0])
        if filepath:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")

    # --- Test 4: Save metadata ---
    print("\n" + "=" * 60)
    print("TEST 4: Save and load metadata")
    print("=" * 60)
    fetcher.save_metadata(papers)
    loaded = ArxivFetcher.load_metadata()
    print(f"  Original: {len(papers)} papers")
    print(f"  Loaded:   {len(loaded)} papers")
    print(f"  Match:    {papers[0].title == loaded[0].title}")

    print("\n✅ All tests passed!")