"""
RAG Chain
---------
Core Retrieval-Augmented Generation pipeline.
Takes a user question, retrieves relevant paper chunks,
and generates a grounded, cited answer.

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
from src.ingestion.vector_store import VectorStore

load_dotenv()


# The system prompt is the most important part of RAG.
# It tells the LLM exactly how to behave.
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
    Core RAG pipeline: Question → Retrieve → Generate → Answer with citations.

    This is the "brain" of the system. It connects the vector store
    (where paper chunks live) with the LLM (which generates answers).
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        top_k: int = 5,
    ):
        """
        Args:
            collection_name: ChromaDB collection to search
            persist_dir: Where ChromaDB data lives
            model_name: Which Groq model to use
            temperature: LLM creativity (0.0 = deterministic, 1.0 = creative).
                        We keep it low (0.1) for factual answers — we want
                        accuracy, not creativity.
            top_k: How many chunks to retrieve per query
        """
        # Initialize vector store (loads embedding model + ChromaDB)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )

        # Initialize the LLM
        # Groq gives us Llama 3.3 70B for free — a very capable model
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
        )

        self.top_k = top_k

        # Build the prompt template
        # ChatPromptTemplate creates a structured prompt with system + user messages
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ])

        # Output parser — extracts the string content from the LLM response
        self.output_parser = StrOutputParser()

        # Build the LCEL chain: prompt → LLM → parse output
        # Note: retrieval is NOT in the chain because we need to
        # format the retrieved chunks before passing to the prompt
        self.chain = self.prompt | self.llm | self.output_parser

        print("RAG Chain initialized.")

    def _retrieve(self, query: str) -> list[dict]:
        """
        Retrieve relevant chunks from the vector store.

        Args:
            query: User's question

        Returns:
            List of retrieved chunks with metadata
        """
        results = self.vector_store.search(query, top_k=self.top_k)
        return results

    def _format_context(self, retrieved_chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a string for the prompt.

        This is crucial — how you present the sources to the LLM
        directly affects the quality of its answer. We include:
        - Source number for easy reference
        - Paper title and page (so the LLM can cite properly)
        - The actual content

        Args:
            retrieved_chunks: List of chunks from vector store search

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant sources found."

        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk["metadata"]
            context_parts.append(
                f"[Source {i}] "
                f"Paper: \"{meta['paper_title']}\" | "
                f"Page: {meta['page_number']} | "
                f"Authors: {meta.get('authors', 'Unknown')}\n"
                f"{chunk['content']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _format_sources(self, retrieved_chunks: list[dict]) -> list[dict]:
        """
        Create a clean list of sources used in the answer.

        This is returned alongside the answer so the user can
        verify claims and read the original papers.
        """
        sources = []
        seen = set()  # Avoid duplicate papers in source list

        for chunk in retrieved_chunks:
            meta = chunk["metadata"]
            paper_id = meta["paper_id"]

            if paper_id not in seen:
                seen.add(paper_id)
                sources.append({
                    "paper_id": paper_id,
                    "title": meta["paper_title"],
                    "authors": meta.get("authors", "Unknown"),
                    "page": meta["page_number"],
                    "distance": chunk["distance"],
                })

        return sources

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get a grounded, cited answer.

        This is the main method. It:
        1. Retrieves relevant chunks
        2. Formats them as context
        3. Sends to LLM with our carefully crafted prompt
        4. Returns the answer with sources

        Args:
            question: Natural language question
            top_k: Override default number of chunks to retrieve

        Returns:
            Dictionary with:
                - answer: The LLM's response (with citations)
                - sources: List of papers used
                - num_chunks_retrieved: How many chunks were used
                - question: The original question (for logging)
        """
        k = top_k or self.top_k

        # Step 1: Retrieve
        print(f"\n🔍 Retrieving top {k} chunks...")
        retrieved = self._retrieve(question)
        print(f"   Found {len(retrieved)} chunks from {len(set(r['metadata']['paper_id'] for r in retrieved))} papers")

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
            "question": question,
        }

    def query_with_details(self, question: str, top_k: int = None) -> dict:
        """
        Like query(), but also returns the raw retrieved chunks.
        Useful for debugging and evaluation.
        """
        k = top_k or self.top_k
        retrieved = self._retrieve(question)
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

    # Test with different types of questions
    questions = [
        "How does LoRA reduce memory usage during fine-tuning of large language models?",
        "What is the attention mechanism and why is it important in transformers?",
        "How does RAG improve the factual accuracy of language model outputs?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = rag.query(question)

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for src in result["sources"]:
            print(f"  - {src['title'][:60]}... (page {src['page']})")

        print(f"\n{'='*60}")