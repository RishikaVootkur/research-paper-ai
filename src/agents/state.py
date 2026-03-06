"""
Agent State
-----------
Defines the shared state that flows through the multi-agent system.

In LangGraph, state is a TypedDict that every node (agent) can read
and write to. Think of it as a shared clipboard that gets passed
from agent to agent.
"""

from typing import TypedDict, Literal, Optional


class AgentState(TypedDict):
    """
    Shared state for the multi-agent system.

    Every agent in the graph receives this state, can read from it,
    and returns an updated version. LangGraph handles merging.
    """
    # --- Input ---
    question: str                     # Original user question

    # --- Router output ---
    route: str                        # "retriever", "synthesizer", or "general"
    question_type: str                # "factual", "comparison", "summary", "methodology"

    # --- Retrieval ---
    retrieved_chunks: list            # Raw retrieved chunks
    context: str                      # Formatted context string
    num_papers: int                   # Number of unique papers retrieved
    search_query: str                 # Actual query used for search (may differ from question)
    hyde_used: bool                   # Whether HyDE was applied

    # --- Generation ---
    answer: str                       # Generated answer
    sources: list                     # Source papers used

    # --- Critic ---
    critique: str                     # Critic's assessment
    quality_score: int                # 1-5 quality rating
    needs_revision: bool              # Whether the answer needs another pass
    revision_count: int               # How many times we've revised (prevent infinite loops)

    # --- Metadata ---
    agent_trace: list                 # Log of which agents ran and what they did
    error: Optional[str]              # Error message if something went wrong