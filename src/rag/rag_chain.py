"""
RAG Chain
---------
Core Retrieval-Augmented Generation pipeline.
Takes a user question, retrieves relevant paper chunks,
and generates a grounded, cited answer.

Now uses HybridRetriever (vector + BM25) for better retrieval.

Usage:
    from src.rag.rag_chain import RAGChain
    rag = RAGChain()
    response = rag.query("How does LoRA reduce memory during fine-tuning?")
    print(response["answer"])
    print(response["sources"])
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.hybrid_retriever import HybridRetriever, RetrievedChunk

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
    Core RAG pipeline with hybrid retrieval.
    Question → Hybrid Retrieve → Generate → Answer with citations.
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        top_k: int = 5,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Args:
            collection_name: ChromaDB collection to search
            persist_dir: Where ChromaDB data lives
            model_name: Which Groq model to use
            temperature: LLM creativity (low = factual)
            top_k: How many chunks to retrieve per query
            vector_weight: Weight for semantic search in hybrid
            bm25_weight: Weight for keyword search in hybrid
        """
        # Initialize hybrid retriever (replaces plain vector store)
        self.retriever = HybridRetriever(
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

        # Build the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ])

        # Build the LCEL chain
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

        print("RAG Chain initialized (with hybrid retrieval).")

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
                f"Authors: {chunk.authors} | "
                f"Match: {chunk.retrieval_method}\n"
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
                    "retrieval_method": chunk.retrieval_method,
                    "score": chunk.score,
                })

        return sources

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get a grounded, cited answer.

        Args:
            question: Natural language question
            top_k: Override default number of chunks to retrieve

        Returns:
            Dictionary with answer, sources, and metadata
        """
        k = top_k or self.top_k

        # Step 1: Hybrid retrieve
        print(f"\n🔍 Hybrid retrieving top {k} chunks...")
        retrieved = self.retriever.search(question, top_k=k)

        # Count unique papers and retrieval methods
        paper_ids = set(r.paper_id for r in retrieved)
        methods = [r.retrieval_method for r in retrieved]
        hybrid_count = methods.count("hybrid")

        print(f"   Found {len(retrieved)} chunks from {len(paper_ids)} papers")
        print(f"   Retrieval methods: {hybrid_count} hybrid, "
              f"{methods.count('vector_only')} vector-only, "
              f"{methods.count('bm25_only')} bm25-only")

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
        retrieved = self.retriever.search(question, top_k=k)
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
        "How does the attention mechanism work in transformers?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = rag.query(question)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES ({result['num_papers']} papers):")
        for src in result["sources"]:
            print(f"  - [{src['retrieval_method']}] {src['title'][:55]}... (p.{src['page']})")

        print(f"\n{'='*60}")