"""
Router Agent
------------
Analyzes the user's question and decides which specialist agent
should handle it. Uses the LLM for intelligent routing rather than
simple keyword matching.

Routes to:
- retriever:   Factual questions, specific lookups, "what is X"
- synthesizer: Comparisons, trends, multi-paper analysis
- general:     Greetings, out-of-scope, chitchat
"""

import os
import sys
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.agents.state import AgentState

load_dotenv()


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for a research paper Q&A system. 
Analyze the user's question and classify it into exactly ONE category.

CATEGORIES:
- "retriever": For factual questions, specific lookups, definitions, 
  explaining how something works. Questions that can be answered by 
  finding the right passage in a paper.
  Examples: "What is LoRA?", "How does attention work?", "What dataset 
  was used in the RAG paper?"

- "synthesizer": For comparisons, trend analysis, summarizing across 
  multiple papers, advantages/disadvantages, or any question that requires 
  combining information from multiple sources.
  Examples: "Compare LoRA and full fine-tuning", "What are the main 
  trends in RAG research?", "What are the pros and cons of different 
  attention mechanisms?"

- "general": For greetings, chitchat, questions completely unrelated to 
  ML/AI research papers, or meta-questions about the system itself.
  Examples: "Hello", "What can you do?", "What's the weather?", 
  "Tell me a joke"

Respond with ONLY a JSON object, nothing else:
{{"route": "<category>", "reasoning": "<one sentence explaining why>"}}"""),
    ("human", "Question: {question}"),
])


class RouterAgent:
    """
    Routes questions to the appropriate specialist agent.

    Uses LLM-based classification for intelligent routing.
    Falls back to keyword-based routing if LLM fails.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=150,
        )
        self.chain = ROUTER_PROMPT | self.llm | StrOutputParser()

    def route(self, state: AgentState) -> AgentState:
        """
        Analyze the question and decide routing.

        This is a LangGraph node — it takes state, updates it, returns it.
        """
        question = state["question"]
        trace = state.get("agent_trace", [])

        try:
            # LLM-based routing
            response = self.chain.invoke({"question": question})

            # Parse JSON response
            clean = response.strip()
            # Handle markdown code blocks if LLM wraps in ```json
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
                clean = clean.strip()

            result = json.loads(clean)
            route = result.get("route", "retriever")
            reasoning = result.get("reasoning", "No reasoning provided")

            # Validate route
            valid_routes = ["retriever", "synthesizer", "general"]
            if route not in valid_routes:
                route = "retriever"  # Default fallback

        except Exception as e:
            # Fallback to simple keyword routing
            route = self._keyword_fallback(question)
            reasoning = f"LLM routing failed ({e}), used keyword fallback"

        # Determine question type for prompt selection
        question_type = self._get_question_type(question, route)

        trace.append({
            "agent": "router",
            "route": route,
            "question_type": question_type,
            "reasoning": reasoning,
        })

        return {
            **state,
            "route": route,
            "question_type": question_type,
            "agent_trace": trace,
            "revision_count": state.get("revision_count", 0),
        }

    def _keyword_fallback(self, question: str) -> str:
        """Simple keyword-based routing as fallback."""
        q = question.lower()

        # General / chitchat
        general_words = ["hello", "hi ", "hey", "thanks", "thank you",
                        "what can you do", "who are you", "weather",
                        "joke", "how are you"]
        if any(w in q for w in general_words):
            return "general"

        # Synthesizer (comparison/trend)
        synth_words = ["compare", "difference", "vs", "versus", "trend",
                      "advantages", "disadvantages", "pros and cons",
                      "summarize across", "overview of", "state of"]
        if any(w in q for w in synth_words):
            return "synthesizer"

        # Default: retriever
        return "retriever"

    def _get_question_type(self, question: str, route: str) -> str:
        """Map route to question type for prompt selection."""
        if route == "general":
            return "default"

        # Use the classifier from our prompts module
        from src.rag.prompts import classify_question
        return classify_question(question)


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    router = RouterAgent()

    test_questions = [
        "What is LoRA?",
        "How does the attention mechanism work?",
        "Compare LoRA and full fine-tuning approaches",
        "What are the main trends in RAG research?",
        "Hello, what can you do?",
        "What's the weather like?",
        "What are the pros and cons of different retrieval strategies for RAG?",
        "Explain how the transformer architecture processes sequences",
    ]

    print("Router Agent Test")
    print("=" * 60)

    for q in test_questions:
        state = {"question": q, "agent_trace": []}
        result = router.route(state)
        trace = result["agent_trace"][-1]
        print(f"\n  Q: {q}")
        print(f"  → Route: {trace['route']} | Type: {trace['question_type']}")
        print(f"    Reason: {trace['reasoning']}")