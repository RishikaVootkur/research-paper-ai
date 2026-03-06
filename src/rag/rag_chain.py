"""
RAG Chain
---------
Core Retrieval-Augmented Generation pipeline.

Full pipeline:
    Question → Classify → Hybrid Search → Re-rank → Select Prompt → LLM → Cited Answer

Usage:
    from src.rag.rag_chain import RAGChain
    rag = RAGChain()
    response = rag.query("How does LoRA reduce memory during fine-tuning?")
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.reranker import RerankedRetriever, RetrievedChunk
from src.rag.prompts import (
    classify_question,
    get_prompt,
    format_context,
    format_sources_list,
)

load_dotenv()


class RAGChain:
    """
    Production-quality RAG pipeline.

    Retrieval: Hybrid Search (vector + BM25) → Cross-Encoder Re-rank
    Generation: Question-type-aware prompts → Groq LLM → Cited answer
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
        # Initialize retrieval pipeline
        self.retriever = RerankedRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # Initialize LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
        )
        self.output_parser = StrOutputParser()

        self.top_k = top_k
        self.fetch_k = fetch_k

        print("RAG Chain initialized (hybrid + re-ranking + smart prompts).")

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get a grounded, cited answer.

        Pipeline:
        1. Classify question type
        2. Retrieve + re-rank relevant chunks
        3. Select appropriate prompt template
        4. Generate answer with LLM
        5. Return answer with sources

        Args:
            question: Natural language question
            top_k: Override default number of chunks

        Returns:
            Dictionary with answer, sources, question type, and metadata
        """
        k = top_k or self.top_k

        # Step 1: Classify question
        question_type = classify_question(question)
        print(f"\n📋 Question type: {question_type}")

        # Step 2: Retrieve and re-rank
        print(f"🔍 Retrieving (hybrid + re-ranking)...")
        retrieved = self.retriever.search(question, top_k=k, fetch_k=self.fetch_k)

        paper_ids = set(r.paper_id for r in retrieved)
        print(f"   {len(retrieved)} chunks from {len(paper_ids)} papers")

        # Step 3: Select prompt and build chain
        prompt = get_prompt(question_type)
        chain = prompt | self.llm | self.output_parser

        # Step 4: Format context and generate
        context = format_context(retrieved)

        print("🤖 Generating answer...")
        answer = chain.invoke({
            "context": context,
            "question": question,
        })

        # Step 5: Format sources
        sources = format_sources_list(retrieved)

        return {
            "answer": answer,
            "sources": sources,
            "question_type": question_type,
            "num_chunks": len(retrieved),
            "num_papers": len(paper_ids),
            "question": question,
        }

    def query_with_details(self, question: str, top_k: int = None) -> dict:
        """Like query(), but also returns retrieved chunks and context."""
        k = top_k or self.top_k
        question_type = classify_question(question)
        retrieved = self.retriever.search(question, top_k=k, fetch_k=self.fetch_k)
        context = format_context(retrieved)

        prompt = get_prompt(question_type)
        chain = prompt | self.llm | self.output_parser
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": format_sources_list(retrieved),
            "retrieved_chunks": retrieved,
            "formatted_context": context,
            "question_type": question_type,
            "question": question,
        }


# ============================================================
# Test with different question types
# ============================================================
if __name__ == "__main__":
    rag = RAGChain()

    questions = [
        # Methodology question
        "How does the attention mechanism work in transformers?",
        # Comparison question
        "What are the advantages of LoRA compared to full fine-tuning?",
        # Summary question
        "What are the main evaluation metrics used for RAG systems?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = rag.query(question)

        print(f"\n📝 Type: {result['question_type']}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES ({result['num_papers']} papers):")
        for src in result["sources"]:
            print(f"  - {src['title'][:55]}... (p.{src['page']})")