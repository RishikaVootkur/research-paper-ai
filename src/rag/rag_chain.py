"""
RAG Chain
---------
Core Retrieval-Augmented Generation pipeline.
Takes a user question, retrieves relevant paper chunks,
and generates a grounded, cited answer.

Retrieval pipeline: Hybrid Search (vector + BM25) → Cross-Encoder Re-rank → LLM

Usage:
    from src.rag.rag_chain import RAGChain
    rag = RAGChain()
    response = rag.query("How does LoRA reduce memory during fine-tuning?")
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.reranker import RerankedRetriever, RetrievedChunk

load_dotenv()


RAG_SYSTEM_PROMPT = """You are a research assistant that answers questions based ONLY on the provided source material from academic papers.

RULES:
1. Answer the question using ONLY the information in the provided sources.
2. After every claim, cite the source using [Paper Title, Page X] format.
3. If the sources don't contain enough information to answer, say "Based on the available sources, I cannot fully answer this question" and explain what you CAN say.
4. Do NOT make up information or use knowledge outside of the provided sources.
5. If sources from different papers disagree, mention both viewpoints.
6. Be specific and technical — this is for researchers, not casual readers.
7. Structure your answer clearly with logical flow."""

RAG_USER_PROMPT = """SOURCES:
{context}

QUESTION: {question}

Provide a detailed, well-cited answer based on the sources above."""


class RAGChain:
    """
    Core RAG pipeline with hybrid retrieval and cross-encoder re-ranking.
    Question → Hybrid Search (20 candidates) → Re-rank (top 5) → LLM → Cited Answer
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        top_k: int = 5,
        fetch_k: int = 20,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Args:
            collection_name: ChromaDB collection to search
            persist_dir: Where ChromaDB data lives
            model_name: Which Groq model to use
            temperature: LLM creativity (low = factual)
            top_k: Final number of chunks after re-ranking
            fetch_k: Number of chunks to fetch before re-ranking
            vector_weight: Weight for semantic search in hybrid
            bm25_weight: Weight for keyword search in hybrid
        """
        # Initialize the full retrieval pipeline (hybrid + reranker)
        self.retriever = RerankedRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # Initialize the LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
        )

        self.top_k = top_k
        self.fetch_k = fetch_k

        # Build the prompt template and chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ])
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

        print("RAG Chain initialized (hybrid retrieval + re-ranking).")

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a string for the prompt."""
        if not chunks:
            return "No relevant sources found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}] "
                f"Paper: \"{chunk.paper_title}\" | "
                f"Page: {chunk.page_number} | "
                f"Authors: {chunk.authors}\n"
                f"{chunk.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _format_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Create a clean list of unique sources used."""
        sources = []
        seen = set()

        for chunk in chunks:
            if chunk.paper_id not in seen:
                seen.add(chunk.paper_id)
                sources.append({
                    "paper_id": chunk.paper_id,
                    "title": chunk.paper_title,
                    "authors": chunk.authors,
                    "page": chunk.page_number,
                    "reranker_score": chunk.score,
                })

        return sources

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get a grounded, cited answer.

        Pipeline: Hybrid Search (fetch 20) → Re-rank (keep 5) → LLM → Answer

        Args:
            question: Natural language question
            top_k: Override default number of final chunks

        Returns:
            Dictionary with answer, sources, and metadata
        """
        k = top_k or self.top_k

        # Step 1: Retrieve and re-rank
        print(f"\n🔍 Retrieving (hybrid + re-ranking)...")
        retrieved = self.retriever.search(question, top_k=k, fetch_k=self.fetch_k)

        paper_ids = set(r.paper_id for r in retrieved)
        print(f"   {len(retrieved)} chunks from {len(paper_ids)} papers (re-ranked from {self.fetch_k} candidates)")

        # Step 2: Format context
        context = self._format_context(retrieved)

        # Step 3: Generate answer
        print("🤖 Generating answer...")
        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        # Step 4: Format sources
        sources = self._format_sources(retrieved)

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": len(retrieved),
            "num_papers": len(paper_ids),
            "question": question,
        }

    def query_with_details(self, question: str, top_k: int = None) -> dict:
        """Like query(), but also returns raw retrieved chunks for debugging."""
        k = top_k or self.top_k
        retrieved = self.retriever.search(question, top_k=k, fetch_k=self.fetch_k)
        context = self._format_context(retrieved)

        answer = self.chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "answer": answer,
            "sources": self._format_sources(retrieved),
            "retrieved_chunks": retrieved,
            "formatted_context": context,
            "question": question,
        }


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    rag = RAGChain()

    questions = [
        "How does LoRA reduce memory usage during fine-tuning of large language models?",
        "What are the main evaluation metrics used for RAG systems?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = rag.query(question)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES ({result['num_papers']} papers):")
        for src in result["sources"]:
            print(f"  - {src['title'][:55]}... (p.{src['page']}, score: {src['reranker_score']:.2f})")