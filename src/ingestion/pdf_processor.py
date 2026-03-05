"""
PDF Processor
-------------
Extracts text from research paper PDFs and splits them into
chunks suitable for embedding and retrieval.
"""

import os
import json
import fitz  # PyMuPDF — the library is called 'fitz' for historical reasons
from dataclasses import dataclass, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class PaperChunk:
    """
    Represents one chunk of text from a paper.

    Each chunk carries metadata about where it came from.
    This is critical — when the RAG system retrieves a chunk,
    we need to know which paper and which section it's from
    so we can cite it properly.
    """
    chunk_id: str          # Unique ID like "2312.10997_chunk_3"
    paper_id: str          # Which paper this came from
    paper_title: str       # Paper title (for citations)
    content: str           # The actual text content
    page_number: int       # Which page this text is from
    chunk_index: int       # Position of this chunk in the paper (0, 1, 2...)
    total_chunks: int      # Total chunks for this paper (filled after splitting)
    char_count: int        # Number of characters in this chunk
    authors: list[str]     # Paper authors (for citations)


class PDFProcessor:
    """
    Handles PDF text extraction and chunking.

    Usage:
        processor = PDFProcessor()
        text = processor.extract_text("path/to/paper.pdf")
        chunks = processor.chunk_text(text, paper_id="2312.10997", paper_title="...")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size: Target size of each chunk in characters (~250 tokens).
                        1000 chars ≈ 200-250 tokens, which is a good balance
                        between having enough context and being specific enough
                        for retrieval.
            chunk_overlap: How many characters overlap between consecutive chunks.
                          200 chars ensures we don't lose context at boundaries.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize the text splitter
        # The separators list defines the "splitting priority":
        # Try double newline first (paragraph break), then single newline,
        # then sentence end, then space, then character-level as last resort
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def extract_text(self, pdf_path: str) -> dict:
        """
        Extract text from a PDF file, organized by page.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with:
                - "full_text": All text combined
                - "pages": List of {page_number, text} dicts
                - "total_pages": Number of pages
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        pages = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Basic cleaning
            text = self._clean_text(text)

            if text.strip():  # Only include pages with actual content
                pages.append({
                    "page_number": page_num + 1,  # 1-indexed for readability
                    "text": text
                })
                full_text_parts.append(text)

        doc.close()

        full_text = "\n\n".join(full_text_parts)

        print(f"  Extracted {len(pages)} pages, {len(full_text)} characters")
        return {
            "full_text": full_text,
            "pages": pages,
            "total_pages": len(pages),
        }

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing common PDF artifacts.

        PDFs are messy — they can have weird characters, excessive
        whitespace, headers/footers repeated on every page, etc.
        This method handles the most common issues.
        """
        # Replace common problematic characters
        text = text.replace("\x00", "")      # Null bytes
        text = text.replace("\ufeff", "")     # BOM (byte order mark)

        # Normalize whitespace: collapse multiple spaces into one
        # but preserve paragraph breaks (double newlines)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = " ".join(line.split())  # Collapse multiple spaces
            cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)

        # Remove excessive newlines (more than 2 in a row)
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text.strip()

    def create_chunks(
        self,
        extracted: dict,
        paper_id: str,
        paper_title: str,
        authors: list[str] = None,
    ) -> list[PaperChunk]:
        """
        Split extracted text into chunks with metadata.

        This is where the magic happens — we take the full paper text
        and split it into retrieval-friendly pieces.

        Args:
            extracted: Output from extract_text()
            paper_id: ArXiv paper ID
            paper_title: Title of the paper
            authors: List of author names

        Returns:
            List of PaperChunk objects
        """
        if authors is None:
            authors = []

        full_text = extracted["full_text"]
        pages = extracted["pages"]

        # Use LangChain's splitter to create chunks
        text_chunks = self.text_splitter.split_text(full_text)

        # Now we need to figure out which page each chunk came from.
        # We do this by tracking character positions.
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Find which page this chunk most likely belongs to
            page_num = self._find_page_number(chunk_text, pages)

            chunk = PaperChunk(
                chunk_id=f"{paper_id}_chunk_{i}",
                paper_id=paper_id,
                paper_title=paper_title,
                content=chunk_text,
                page_number=page_num,
                chunk_index=i,
                total_chunks=len(text_chunks),  # Will be same for all chunks of this paper
                char_count=len(chunk_text),
                authors=authors,
            )
            chunks.append(chunk)

        print(f"  Created {len(chunks)} chunks (avg {sum(c.char_count for c in chunks) // len(chunks)} chars each)")
        return chunks

    def _find_page_number(self, chunk_text: str, pages: list[dict]) -> int:
        """
        Determine which page a chunk of text belongs to.

        Simple approach: find the page that has the most overlap
        with the chunk text. We check the first 100 characters
        of the chunk against each page.
        """
        search_text = chunk_text[:100]

        for page in pages:
            if search_text in page["text"]:
                return page["page_number"]

        # If exact match fails, find the page with the most common words
        chunk_words = set(chunk_text[:200].split())
        best_page = 1
        best_overlap = 0

        for page in pages:
            page_words = set(page["text"].split())
            overlap = len(chunk_words & page_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page["page_number"]

        return best_page

    def process_pdf(
        self,
        pdf_path: str,
        paper_id: str,
        paper_title: str,
        authors: list[str] = None,
    ) -> list[PaperChunk]:
        """
        Full pipeline: extract text from PDF and create chunks.
        This is the main method you'll call.

        Args:
            pdf_path: Path to PDF file
            paper_id: ArXiv paper ID
            paper_title: Paper title
            authors: Author names

        Returns:
            List of PaperChunk objects
        """
        print(f"\nProcessing: {paper_title[:60]}...")

        # Step 1: Extract text
        extracted = self.extract_text(pdf_path)

        # Step 2: Create chunks
        chunks = self.create_chunks(extracted, paper_id, paper_title, authors)

        return chunks

    def save_chunks(
        self,
        chunks: list[PaperChunk],
        filepath: str = "data/processed/chunks.json"
    ):
        """Save chunks to JSON for inspection or later use."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = [asdict(chunk) for chunk in chunks]

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(chunks)} chunks to {filepath}")

    @staticmethod
    def load_chunks(filepath: str = "data/processed/chunks.json") -> list[PaperChunk]:
        """Load chunks from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)

        chunks = [PaperChunk(**item) for item in data]
        print(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    from arxiv_fetcher import ArxivFetcher

    fetcher = ArxivFetcher()
    processor = PDFProcessor()

    # Fetch and download a couple of papers to test with
    print("=" * 60)
    print("STEP 1: Fetching papers")
    print("=" * 60)
    papers = fetcher.search_by_ids([
        "2005.11401",  # RAG original paper (smaller, good for testing)
    ])

    print("\n" + "=" * 60)
    print("STEP 2: Downloading PDFs")
    print("=" * 60)
    downloaded = fetcher.download_papers(papers)

    print("\n" + "=" * 60)
    print("STEP 3: Extracting text and creating chunks")
    print("=" * 60)
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

    # Show some example chunks
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)
    for chunk in all_chunks[:3]:  # Show first 3 chunks
        print(f"\n--- {chunk.chunk_id} (page {chunk.page_number}) ---")
        print(f"Characters: {chunk.char_count}")
        print(f"Content preview: {chunk.content[:300]}...")

    # Save chunks
    print("\n" + "=" * 60)
    print("STEP 4: Saving chunks")
    print("=" * 60)
    processor.save_chunks(all_chunks)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Papers processed: {len(papers)}")
    print(f"Total chunks: {len(all_chunks)}")
    avg_size = sum(c.char_count for c in all_chunks) / len(all_chunks) if all_chunks else 0
    print(f"Average chunk size: {avg_size:.0f} characters")
    print(f"Chunk size setting: {processor.chunk_size}")
    print(f"Chunk overlap setting: {processor.chunk_overlap}")

    print("\n✅ PDF processing complete!")