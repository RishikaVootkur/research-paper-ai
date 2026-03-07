"""
Prompt Manager
--------------
Manages prompts for different types of RAG queries.
Ensures consistent citation formatting and answer quality.

The prompt is the single biggest lever for answer quality.
Good retrieval + bad prompt = bad answer.
Good retrieval + good prompt = great answer.
"""

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Citation format instruction (shared across all prompts)
# ============================================================
CITATION_INSTRUCTIONS = """CITATION FORMAT (you MUST follow this exactly):
- Cite every claim using this format: [1], [2], etc. matching the source numbers.
- At the END of your answer, include a "References" section listing each source you cited:
  [1] "Paper Title" — Authors (Page X)
  [2] "Paper Title" — Authors (Page X)
- Only include sources you actually referenced in your answer.
- If you cannot answer from the sources, state that clearly.

CRITICAL GROUNDING RULES:
- NEVER add information that is not explicitly stated in the sources.
- If a source says "reduces parameters by 10,000x", say exactly that — do not round or paraphrase loosely.
- Do NOT use your general knowledge to fill gaps. If the sources don't mention something, don't mention it.
- It is better to give a shorter, fully grounded answer than a longer answer with unsupported claims.
- Every single sentence in your answer must be traceable to a specific source."""


# ============================================================
# System prompts for different question types
# ============================================================

FACTUAL_SYSTEM_PROMPT = f"""You are a precise research assistant. Answer the question using ONLY the provided sources.

{CITATION_INSTRUCTIONS}

STYLE GUIDELINES:
- Be direct and specific. No filler or unnecessary introductions.
- Start with the core answer, then add supporting details.
- Use technical language appropriate for a researcher audience.
- If sources provide specific numbers, methods, or results, include them."""

COMPARISON_SYSTEM_PROMPT = f"""You are a research assistant specializing in comparative analysis. Compare the concepts/methods using ONLY the provided sources.

{CITATION_INSTRUCTIONS}

STYLE GUIDELINES:
- Structure your answer with clear sections for each item being compared.
- Highlight key differences and similarities.
- If the sources show one approach outperforming another, state the evidence.
- End with a brief synthesis of the comparison."""

SUMMARY_SYSTEM_PROMPT = f"""You are a research assistant that creates concise summaries. Summarize the topic using ONLY the provided sources.

{CITATION_INSTRUCTIONS}

STYLE GUIDELINES:
- Start with a high-level overview (2-3 sentences).
- Then cover the key points systematically.
- Group related information together.
- Keep it concise — no repetition."""

METHODOLOGY_SYSTEM_PROMPT = f"""You are a research assistant explaining technical methods. Explain the methodology using ONLY the provided sources.

{CITATION_INSTRUCTIONS}

STYLE GUIDELINES:
- Explain step by step how the method works.
- Include mathematical formulations or algorithms if mentioned in sources.
- Note any assumptions, limitations, or prerequisites.
- If applicable, mention how this method compares to alternatives."""

DEFAULT_SYSTEM_PROMPT = f"""You are a knowledgeable research assistant. Answer the question using ONLY the provided sources.

{CITATION_INSTRUCTIONS}

STYLE GUIDELINES:
- Be thorough but concise.
- Cite every factual claim.
- Structure your answer logically.
- If the sources don't fully address the question, state what you can and cannot answer."""


# ============================================================
# User prompt template (same for all types)
# ============================================================

USER_PROMPT = """SOURCES:
{context}

QUESTION: {question}

Answer based on the sources above, following the citation format exactly."""


# ============================================================
# Question type classifier (simple keyword-based)
# ============================================================

def classify_question(question: str) -> str:
    """
    Classify a question into a type to select the best prompt.

    This is a simple keyword-based classifier. In Week 3, our
    Router Agent will do this more intelligently using an LLM.
    For now, keywords work well enough.

    Returns:
        One of: "factual", "comparison", "summary", "methodology", "default"
    """
    q = question.lower()

    # Comparison indicators
    comparison_words = ["compare", "difference", "vs", "versus", "differ",
                       "contrast", "advantage", "disadvantage", "better",
                       "worse", "trade-off", "tradeoff"]
    if any(word in q for word in comparison_words):
        return "comparison"

    # Summary indicators
    summary_words = ["summarize", "summary", "overview", "survey",
                    "what are the main", "what are the key",
                    "trends", "landscape", "state of"]
    if any(word in q for word in summary_words):
        return "summary"

    # Methodology indicators
    method_words = ["how does", "how do", "how is", "how are",
                   "explain", "describe the process", "mechanism",
                   "algorithm", "method", "approach", "technique",
                   "step by step", "pipeline", "architecture"]
    if any(word in q for word in method_words):
        return "methodology"

    # Factual indicators (what, when, who, which)
    factual_words = ["what is", "what are", "who", "when", "which",
                    "define", "definition", "name", "list"]
    if any(word in q for word in factual_words):
        return "factual"

    return "default"


def get_prompt(question_type: str) -> ChatPromptTemplate:
    """
    Get the appropriate prompt template for a question type.

    Args:
        question_type: Output from classify_question()

    Returns:
        ChatPromptTemplate configured for that question type
    """
    system_prompts = {
        "factual": FACTUAL_SYSTEM_PROMPT,
        "comparison": COMPARISON_SYSTEM_PROMPT,
        "summary": SUMMARY_SYSTEM_PROMPT,
        "methodology": METHODOLOGY_SYSTEM_PROMPT,
        "default": DEFAULT_SYSTEM_PROMPT,
    }

    system_prompt = system_prompts.get(question_type, DEFAULT_SYSTEM_PROMPT)

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", USER_PROMPT),
    ])


# ============================================================
# Context formatter (improved from what we had in rag_chain.py)
# ============================================================

def format_context(chunks) -> str:
    """
    Format retrieved chunks into a clean, numbered context string.

    The format is designed to make it easy for the LLM to cite:
    each source has a clear number, title, author, and page.
    """
    if not chunks:
        return "No relevant sources found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}]\n"
            f"Paper: \"{chunk.paper_title}\"\n"
            f"Authors: {chunk.authors}\n"
            f"Page: {chunk.page_number}\n"
            f"Content:\n{chunk.content}"
        )

    return "\n\n" + "=" * 40 + "\n\n".join(parts)


def format_sources_list(chunks) -> list[dict]:
    """Create a deduplicated list of source papers."""
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
                "score": chunk.score,
            })

    return sources


# ============================================================
# Test the classifier
# ============================================================
if __name__ == "__main__":
    test_questions = [
        "What is LoRA?",
        "How does the attention mechanism work in transformers?",
        "Compare LoRA and full fine-tuning for large language models",
        "What are the main evaluation metrics for RAG systems?",
        "Summarize the key trends in retrieval augmented generation",
        "Explain the architecture of the transformer model step by step",
        "What are the advantages of RAG over pure parametric models?",
        "Who introduced the concept of retrieval augmented generation?",
    ]

    print("Question Type Classification")
    print("=" * 60)
    for q in test_questions:
        qtype = classify_question(q)
        print(f"  [{qtype:12s}] {q}")