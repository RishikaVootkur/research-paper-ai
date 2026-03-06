"""
Interactive Multi-Agent Demo
-----------------------------
A terminal-based chat interface powered by the multi-agent system.
Routes questions to specialized agents and shows the agent trace.

Run: python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from src.agents.graph import AgentOrchestrator


def print_banner():
    print("\n" + "=" * 60)
    print("  📚 Research Paper Intelligence Platform")
    print("  Multi-Agent System powered by LangGraph")
    print("=" * 60)
    print("\nAgents:")
    print("  🔀 Router      — classifies your question")
    print("  🔍 Retriever   — factual lookups from papers")
    print("  🔬 Synthesizer — multi-paper comparison & analysis")
    print("  💬 General     — greetings & general help")
    print("  ✅ Critic      — quality checks every answer")
    print("\nCommands:")
    print("  /trace   — Show agent trace for last answer")
    print("  /stats   — Show database stats")
    print("  /quit    — Exit")
    print("=" * 60)


def main():
    print("\nLoading multi-agent system (this takes a few seconds)...")
    orchestrator = AgentOrchestrator()

    # Show database stats
    stats = orchestrator.retriever_agent.retriever.hybrid_retriever.vector_store.get_collection_stats()
    print(f"\n📊 Database: {stats['total_chunks']} chunks from {stats['papers']} papers")

    print_banner()

    last_result = None

    while True:
        try:
            question = input("\n💬 You: ").strip()

            if not question:
                continue

            if question.startswith("/"):
                cmd = question.lower()

                if cmd == "/quit":
                    print("\nGoodbye! 👋")
                    break

                elif cmd == "/trace":
                    if last_result:
                        orchestrator.print_trace(last_result)
                    else:
                        print("No previous answer to show trace for.")
                    continue

                elif cmd == "/stats":
                    s = orchestrator.retriever_agent.retriever.hybrid_retriever.vector_store.get_collection_stats()
                    print(f"  Chunks: {s['total_chunks']}")
                    print(f"  Papers: {s['papers']}")
                    continue

                else:
                    print("Unknown command. Try /trace, /stats, or /quit")
                    continue

            # Run through the agent system
            result = orchestrator.run(question)
            last_result = result

            # Display answer
            print(f"\n🤖 Assistant:\n")
            print(result["answer"])

            # Display metadata bar
            agent_icon = {"retriever": "🔍", "synthesizer": "🔬", "general": "💬"}.get(result["route"], "❓")
            print(f"\n{agent_icon} Agent: {result['route']} "
                  f"| Type: {result['question_type']} "
                  f"| Quality: {result['quality_score']}/5 "
                  f"| Papers: {result['num_papers']}"
                  f"{' | HyDE' if result.get('hyde_used') else ''}")

            if result["sources"]:
                print(f"📎 Sources:")
                for src in result["sources"]:
                    print(f"   - {src['title'][:60]}...")

            print(f"\n   (type /trace to see full agent trace)")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type /quit to exit.")


if __name__ == "__main__":
    main()