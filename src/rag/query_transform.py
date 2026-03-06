"""
Query Transformer
-----------------
Transforms user queries to improve retrieval quality.

Techniques:
1. HyDE (Hypothetical Document Embeddings): Generate a hypothetical
   answer, embed it, and use it for retrieval.
2. Query Expansion: Expand the query with related technical terms.

These techniques bridge the gap between casual user language
and technical paper language.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ============================================================
# HyDE: Hypothetical Document Embeddings
# ============================================================

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research paper author. Given a question, write a SHORT paragraph 
(3-5 sentences) that would appear in a research paper answering this question.

RULES:
- Write in academic/technical style, as if from a real paper.
- Use specific technical terminology that a paper on this topic would use.
- Include relevant method names, metrics, or concepts.
- Do NOT say "this paper" or "we propose" — just state the facts.
- Keep it to ONE paragraph, 3-5 sentences max."""),
    ("human", "Question: {question}\n\nHypothetical paper paragraph:"),
])


# ============================================================
# Query Expansion
# ============================================================

EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a search query optimizer for academic papers. 
Given a user's question, generate an improved search query that will 
find the most relevant research papers.

RULES:
- Add key technical terms and synonyms the user might have missed.
- Keep the expanded query concise (under 30 words).
- Include both the original intent and related technical terms.
- Output ONLY the expanded query, nothing else."""),
    ("human", "Original question: {question}\n\nExpanded search query:"),
])


class QueryTransformer:
    """
    Transforms queries to improve retrieval quality.

    Provides two strategies:
    - HyDE: Generate a hypothetical answer, search with its embedding
    - Expansion: Add technical terms to the query

    Usage:
        transformer = QueryTransformer()
        hyde_text = transformer.hyde("How do you stop LLMs from making things up?")
        # Returns a technical paragraph about hallucination reduction
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.3,  # Slightly creative for generating hypothetical docs
            max_tokens=300,
        )
        self.output_parser = StrOutputParser()

        self.hyde_chain = HYDE_PROMPT | self.llm | self.output_parser
        self.expansion_chain = EXPANSION_PROMPT | self.llm | self.output_parser

        print("Query Transformer initialized.")

    def hyde(self, question: str) -> str:
        """
        Generate a hypothetical document passage for the question.

        The returned text is meant to be EMBEDDED and used for search,
        not shown to the user. It bridges the vocabulary gap between
        user language and paper language.

        Args:
            question: User's question

        Returns:
            Hypothetical paper paragraph answering the question
        """
        result = self.hyde_chain.invoke({"question": question})
        return result.strip()

    def expand(self, question: str) -> str:
        """
        Expand the query with related technical terms.

        Lighter-weight than HyDE — just adds relevant keywords
        without generating a full paragraph.

        Args:
            question: User's question

        Returns:
            Expanded search query
        """
        result = self.expansion_chain.invoke({"question": question})
        return result.strip()

    def should_use_hyde(self, question: str) -> bool:
        """
        Decide whether HyDE would help for this question.

        HyDE helps with casual/vague queries but can hurt with
        precise keyword queries. Simple heuristic:
        - Short, casual questions → use HyDE
        - Technical questions with specific terms → skip HyDE
        """
        q = question.lower()

        # Indicators that the query is already technical/specific
        # If the query contains specific names, acronyms, or technical terms,
        # the raw query is better than a hypothetical passage
        technical_indicators = [
            "arxiv", "paper", "et al", "algorithm",
            "equation", "formula", "theorem", "proof",
            "table", "figure", "section", "appendix",
            "ragas", "lora", "rlhf", "qlora", "peft",
            "bert", "gpt", "llama", "mistral", "gemini",
            "framework", "benchmark", "dataset", "metric",
        ]
        if any(term in q for term in technical_indicators):
            return False

        # Indicators that HyDE would help (casual language)
        casual_indicators = [
            "how do you", "why do", "what's the deal",
            "tell me about", "what happens when",
            "is it possible", "making things up",
            "get better results", "stop ai from",
            "really big", "so hard to",
        ]
        if any(term in q for term in casual_indicators):
            return True

        # Default: use HyDE for shorter questions (likely casual),
        # skip for longer ones (likely already detailed)
        word_count = len(question.split())
        return word_count < 10


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    transformer = QueryTransformer()

    test_queries = [
        # Casual / non-technical (HyDE should help)
        "How do you make LLMs stop making things up?",
        "Why is it so hard to train really big models?",
        "Can you explain how search works in AI chatbots?",
        # Technical / specific (HyDE less useful)
        "What is the LoRA rank r parameter and how does it affect adaptation?",
        "Describe the RAGAS evaluation framework for RAG systems",
    ]

    for query in test_queries:
        should_hyde = transformer.should_use_hyde(query)
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"Use HyDE? {should_hyde}")
        print(f"{'='*60}")

        # Always show both for comparison
        print(f"\n📝 EXPANDED QUERY:")
        expanded = transformer.expand(query)
        print(f"   {expanded}")

        print(f"\n📄 HYDE (hypothetical passage):")
        hyde = transformer.hyde(query)
        print(f"   {hyde}")