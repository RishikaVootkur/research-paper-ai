"""
Agent Orchestration Graph
-------------------------
Wires all agents together using LangGraph.

Flow:
    User Query → Router → [Retriever | Synthesizer | General] → Critic → Final Answer
                                                                   ↓
                                                              (if revision needed)
                                                                   ↓
                                                         Back to specialist agent

Usage:
    from src.agents.graph import AgentOrchestrator
    orchestrator = AgentOrchestrator()
    result = orchestrator.run("How does LoRA work?")
"""

import os
import sys
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.agents.state import AgentState
from src.agents.router import RouterAgent
from src.agents.specialists import RetrieverAgent, SynthesizerAgent, GeneralAgent
from src.agents.critic import CriticAgent
from src.rag.reranker import RerankedRetriever
from src.rag.query_transform import QueryTransformer


class AgentOrchestrator:
    """
    Orchestrates the multi-agent system using LangGraph.

    Builds a state graph where:
    - Nodes are agents (router, retriever, synthesizer, general, critic)
    - Edges define the flow between agents
    - Conditional edges handle routing and revision decisions
    """

    def __init__(
        self,
        collection_name: str = "research_papers",
        persist_dir: str = "chroma_db",
    ):
        print("Initializing Agent Orchestrator...")

        # Shared components
        retriever = RerankedRetriever(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
        query_transformer = QueryTransformer()

        # Initialize all agents
        self.router = RouterAgent()
        self.retriever_agent = RetrieverAgent(retriever, query_transformer)
        self.synthesizer_agent = SynthesizerAgent(retriever, query_transformer)
        self.general_agent = GeneralAgent()
        self.critic = CriticAgent()

        # Build the graph
        self.graph = self._build_graph()
        print("Agent Orchestrator ready.")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine.

        Graph structure:
            START → router → [retriever | synthesizer | general] → critic → END
                                                                      ↓
                                                                 (revision?) → back to specialist
        """
        # Create the graph with our state type
        graph = StateGraph(AgentState)

        # Add nodes (each node is a function that takes state → returns state)
        graph.add_node("router", self.router.route)
        graph.add_node("retriever", self.retriever_agent.run)
        graph.add_node("synthesizer", self.synthesizer_agent.run)
        graph.add_node("general", self.general_agent.run)
        graph.add_node("critic", self.critic.review)

        # Set entry point
        graph.set_entry_point("router")

        # Router → specialist (conditional edge based on route)
        graph.add_conditional_edges(
            "router",
            self._route_to_specialist,
            {
                "retriever": "retriever",
                "synthesizer": "synthesizer",
                "general": "general",
            }
        )

        # Each specialist → critic
        graph.add_edge("retriever", "critic")
        graph.add_edge("synthesizer", "critic")
        graph.add_edge("general", "critic")

        # Critic → END or back to specialist (conditional)
        graph.add_conditional_edges(
            "critic",
            self._should_revise,
            {
                "accept": END,
                "revise_retriever": "retriever",
                "revise_synthesizer": "synthesizer",
            }
        )

        # Compile the graph (makes it executable)
        return graph.compile()

    def _route_to_specialist(self, state: AgentState) -> str:
        """Conditional edge: route to the appropriate specialist."""
        return state.get("route", "retriever")

    def _should_revise(self, state: AgentState) -> str:
        """Conditional edge: accept the answer or send back for revision."""
        if state.get("needs_revision", False):
            route = state.get("route", "retriever")
            if route == "synthesizer":
                return "revise_synthesizer"
            return "revise_retriever"
        return "accept"

    def run(self, question: str) -> dict:
        """
        Run the full multi-agent pipeline.

        Args:
            question: User's question

        Returns:
            Dictionary with answer, sources, agent trace, and quality info
        """
        # Initialize state
        initial_state: AgentState = {
            "question": question,
            "route": "",
            "question_type": "",
            "retrieved_chunks": [],
            "context": "",
            "num_papers": 0,
            "search_query": "",
            "hyde_used": False,
            "answer": "",
            "sources": [],
            "critique": "",
            "quality_score": 0,
            "needs_revision": False,
            "revision_count": 0,
            "agent_trace": [],
            "error": None,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return {
            "question": question,
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "route": final_state.get("route", ""),
            "question_type": final_state.get("question_type", ""),
            "quality_score": final_state.get("quality_score", 0),
            "critique": final_state.get("critique", ""),
            "num_papers": final_state.get("num_papers", 0),
            "hyde_used": final_state.get("hyde_used", False),
            "agent_trace": final_state.get("agent_trace", []),
        }

    def print_trace(self, result: dict):
        """Pretty-print the agent trace for debugging."""
        print(f"\n🔗 AGENT TRACE:")
        for step in result["agent_trace"]:
            agent = step.get("agent", "unknown")
            action = step.get("action", "unknown")
            print(f"   → [{agent}] {action}", end="")

            if agent == "router":
                print(f" → route={step.get('route')} ({step.get('reasoning', '')[:60]})")
            elif agent == "retriever":
                print(f" → {step.get('chunks_retrieved', 0)} chunks, "
                      f"{step.get('papers_found', 0)} papers"
                      f"{', HyDE' if step.get('hyde_used') else ''}")
            elif agent == "synthesizer":
                print(f" → {step.get('chunks_retrieved', 0)} chunks, "
                      f"{step.get('papers_found', 0)} papers")
            elif agent == "critic":
                score = step.get("score", "N/A")
                print(f" → score={score}/5"
                      f"{', REVISION NEEDED' if step.get('needs_revision') else ''}")
            else:
                print()


# ============================================================
# Test it
# ============================================================
if __name__ == "__main__":
    orchestrator = AgentOrchestrator()

    questions = [
        "What is LoRA and how does it reduce memory during fine-tuning?",
        "Compare LoRA and full fine-tuning for large language models",
        "Hello! What can you help me with?",
        "What are the key evaluation metrics for RAG systems?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

        result = orchestrator.run(question)

        print(f"\n📋 Route: {result['route']} | Type: {result['question_type']}")
        print(f"⭐ Quality: {result['quality_score']}/5")
        print(f"\nANSWER:\n{result['answer'][:500]}...")

        if result["sources"]:
            print(f"\nSOURCES ({result['num_papers']} papers):")
            for src in result["sources"]:
                print(f"  - {src['title'][:55]}...")

        orchestrator.print_trace(result)