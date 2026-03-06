"""
RAG Chain
---------
Production-quality Retrieval-Augmented Generation pipeline.

Full pipeline:
    Question → [HyDE if needed] → Classify → Hybrid Search → Re-rank → Prompt → LLM → Cited Answer

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
from src.rag.query_transform import QueryTransformer
from src.rag.prompts import (
    classify_question,
    get_prompt,
    format_context,
    format_sources_list,
)

load_dotenv()


class RAGChain:
    """
    Production-quality RAG pipeline with all advanced features:
    - Hybrid search (vector + BM25)
    - Cross-encoder re-ranking
    - HyDE query transformation
    - Question-type-aware prompts
    - Standardized citations
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
        use_hyde: str = "auto",
    ):
        """
        Args:
            use_hyde: "auto" (let system decide), "always", or "never"
        """
        # Retrieval pipeline
        self.retriever = RerankedRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        # Query transformer (HyDE + expansion)
        self.query_transformer = QueryTransformer()

        # LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=2048,
        )
        self.output_parser = StrOutputParser()

        self.top_k = top_k
        self.fetch_k = fetch_k
        self.use_hyde = use_hyde

        print("RAG Chain initialized (hybrid + re-ranking + HyDE + smart prompts).")

    def _get_search_query(self, question: str) -> dict:
        """
        Decide whether to transform the query and return the search query.

        Returns:
            Dict with "search_query" and "hyde_used" flag
        """
        use_hyde = False

        if self.use_hyde == "always":
            use_hyde = True
        elif self.use_hyde == "auto":
            use_hyde = self.query_transformer.should_use_hyde(question)

        if use_hyde:
            hyde_passage = self.query_transformer.hyde(question)
            # Use the HyDE passage as search query for embedding,
            # but also combine with expanded query for BM25
            expanded = self.query_transformer.expand(question)
            # For hybrid search: HyDE helps vector search, expansion helps BM25
            # We use HyDE passage since it helps the dominant vector component
            return {"search_query": hyde_passage, "hyde_used": True}
        else:
            return {"search_query": question, "hyde_used": False}

    def query(self, question: str, top_k: int = None) -> dict:
        """
        Ask a question and get a grounded, cited answer.

        Full pipeline:
        1. Optionally transform query with HyDE
        2. Classify question type
        3. Hybrid search + re-rank
        4. Select prompt based on question type
        5. Generate answer with LLM

        Args:
            question: Natural language question
            top_k: Override default number of chunks

        Returns:
            Dictionary with answer, sources, and metadata
        """
        k = top_k or self.top_k

        # Step 1: Query transformation
        query_info = self._get_search_query(question)
        search_query = query_info["search_query"]

        if query_info["hyde_used"]:
            print(f"\n🔮 HyDE activated — searching with hypothetical passage")
            print(f"   HyDE passage: {search_query[:100]}...")
        else:
            print(f"\n🔍 Direct search (no HyDE)")

        # Step 2: Classify question type (always use original question)
        question_type = classify_question(question)
        print(f"📋 Question type: {question_type}")

        # Step 3: Retrieve and re-rank
        print(f"🔍 Retrieving (hybrid + re-ranking)...")
        retrieved = self.retriever.search(search_query, top_k=k, fetch_k=self.fetch_k)

        paper_ids = set(r.paper_id for r in retrieved)
        print(f"   {len(retrieved)} chunks from {len(paper_ids)} papers")

        # Step 4: Select prompt and build chain
        prompt = get_prompt(question_type)
        chain = prompt | self.llm | self.output_parser

        # Step 5: Generate answer (always use original question in prompt)
        context = format_context(retrieved)
        print("🤖 Generating answer...")
        answer = chain.invoke({
            "context": context,
            "question": question,  # Original question, not HyDE passage
        })

        # Format sources
        sources = format_sources_list(retrieved)

        return {
            "answer": answer,
            "sources": sources,
            "question_type": question_type,
            "hyde_used": query_info["hyde_used"],
            "num_chunks": len(retrieved),
            "num_papers": len(paper_ids),
            "question": question,
        }

    def query_with_details(self, question: str, top_k: int = None) -> dict:
        """Like query(), but also returns retrieved chunks and context."""
        k = top_k or self.top_k
        query_info = self._get_search_query(question)
        question_type = classify_question(question)
        retrieved = self.retriever.search(
            query_info["search_query"], top_k=k, fetch_k=self.fetch_k
        )
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
            "hyde_used": query_info["hyde_used"],
            "question": question,
        }


# ============================================================
# Test: Compare with and without HyDE
# ============================================================
if __name__ == "__main__":
    # Test with a casual query where HyDE should help
    print("\n" + "=" * 60)
    print("TEST 1: Casual query (HyDE should activate)")
    print("=" * 60)

    rag_hyde = RAGChain(use_hyde="always")
    rag_no_hyde = RAGChain(use_hyde="never")

    casual_query = "How do you stop AI from making things up?"

    print(f"\nQUERY: {casual_query}")
    print("\n--- WITHOUT HyDE ---")
    r1 = rag_no_hyde.query(casual_query)
    print(f"\nSources: {[s['title'][:40] for s in r1['sources']]}")
    print(f"Answer preview: {r1['answer'][:300]}...")

    print("\n--- WITH HyDE ---")
    r2 = rag_hyde.query(casual_query)
    print(f"\nSources: {[s['title'][:40] for s in r2['sources']]}")
    print(f"Answer preview: {r2['answer'][:300]}...")

    # Test with a technical query where HyDE shouldn't be needed
    print("\n" + "=" * 60)
    print("TEST 2: Technical query (auto mode)")
    print("=" * 60)

    rag_auto = RAGChain(use_hyde="auto")
    technical_query = "What is the RAGAS framework and what metrics does it define?"
    result = rag_auto.query(technical_query)
    print(f"\nHyDE used: {result['hyde_used']}")
    print(f"Sources: {[s['title'][:40] for s in result['sources']]}")
    print(f"Answer preview: {result['answer'][:300]}...")