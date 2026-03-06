"""
Agent System Tests
------------------
Tests the multi-agent orchestrator with diverse query types.

Run: python tests/test_agents.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.agents.graph import AgentOrchestrator


def test_routing(orchestrator: AgentOrchestrator):
    """Test that questions get routed to the correct agent."""
    print("\n🔀 TEST: Routing")
    print("-" * 40)

    test_cases = [
        ("What is LoRA?", "retriever"),
        ("Compare LoRA and full fine-tuning", "synthesizer"),
        ("Hello!", "general"),
        ("How does the attention mechanism work?", "retriever"),
        ("What are the pros and cons of RAG vs fine-tuning?", "synthesizer"),
        ("Thanks, goodbye", "general"),
    ]

    passed = 0
    for question, expected_route in test_cases:
        result = orchestrator.run(question)
        actual_route = result["route"]
        status = "✅" if actual_route == expected_route else "❌"
        if actual_route == expected_route:
            passed += 1
        print(f"  {status} '{question[:40]}...' → {actual_route} (expected: {expected_route})")

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed >= len(test_cases) - 1  # Allow 1 routing disagreement


def test_quality_scores(orchestrator: AgentOrchestrator):
    """Test that answers get quality scores >= 3."""
    print("\n⭐ TEST: Quality Scores")
    print("-" * 40)

    questions = [
        "How does LoRA reduce memory usage?",
        "What are the main evaluation metrics for RAG?",
        "Compare different attention mechanisms in transformers",
    ]

    passed = 0
    for question in questions:
        result = orchestrator.run(question)
        score = result["quality_score"]
        status = "✅" if score >= 3 else "❌"
        if score >= 3:
            passed += 1
        print(f"  {status} '{question[:40]}...' → score: {score}/5")

    print(f"\n  Result: {passed}/{len(questions)} passed")
    return passed == len(questions)


def test_agent_trace(orchestrator: AgentOrchestrator):
    """Test that every response has a complete agent trace."""
    print("\n🔗 TEST: Agent Trace Completeness")
    print("-" * 40)

    questions = [
        "What is LoRA?",
        "Hello!",
        "Compare LoRA and full fine-tuning",
    ]

    passed = 0
    for question in questions:
        result = orchestrator.run(question)
        trace = result["agent_trace"]

        # Every trace should have at least: router + specialist + critic
        agents_in_trace = [step["agent"] for step in trace]
        has_router = "router" in agents_in_trace
        has_specialist = any(a in agents_in_trace for a in ["retriever", "synthesizer", "general"])
        has_critic = "critic" in agents_in_trace

        all_present = has_router and has_specialist and has_critic
        status = "✅" if all_present else "❌"
        if all_present:
            passed += 1
        print(f"  {status} '{question[:40]}...' → agents: {agents_in_trace}")

    print(f"\n  Result: {passed}/{len(questions)} passed")
    return passed == len(questions)


def test_synthesizer_multi_paper(orchestrator: AgentOrchestrator):
    """Test that synthesizer pulls from multiple papers."""
    print("\n📚 TEST: Synthesizer Multi-Paper")
    print("-" * 40)

    question = "What are the different approaches to efficient fine-tuning of language models?"
    result = orchestrator.run(question)

    num_papers = result["num_papers"]
    route = result["route"]

    is_synth = route == "synthesizer"
    multi_paper = num_papers >= 2

    status = "✅" if (is_synth and multi_paper) else "❌"
    print(f"  {status} Route: {route}, Papers: {num_papers}")
    print(f"     Sources: {[s['title'][:40] for s in result['sources']]}")

    passed = is_synth and multi_paper
    print(f"\n  Result: {'1/1 passed' if passed else '0/1 passed'}")
    return passed


def test_general_no_sources(orchestrator: AgentOrchestrator):
    """Test that general agent doesn't return fake sources."""
    print("\n💬 TEST: General Agent (no fake sources)")
    print("-" * 40)

    questions = ["Hello!", "What's the weather?", "Tell me a joke"]
    
    passed = 0
    for question in questions:
        result = orchestrator.run(question)
        no_sources = len(result["sources"]) == 0
        status = "✅" if no_sources else "❌"
        if no_sources:
            passed += 1
        print(f"  {status} '{question}' → sources: {len(result['sources'])}")

    print(f"\n  Result: {passed}/{len(questions)} passed")
    return passed == len(questions)


def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 60)
    print("  MULTI-AGENT SYSTEM TEST SUITE")
    print("=" * 60)

    start_time = time.time()

    print("\n⏳ Initializing orchestrator...")
    orchestrator = AgentOrchestrator()

    t1 = test_routing(orchestrator)
    t2 = test_quality_scores(orchestrator)
    t3 = test_agent_trace(orchestrator)
    t4 = test_synthesizer_multi_paper(orchestrator)
    t5 = test_general_no_sources(orchestrator)

    elapsed = time.time() - start_time

    results = [t1, t2, t3, t4, t5]
    names = [
        "Routing Accuracy",
        "Quality Scores (>= 3/5)",
        "Agent Trace Completeness",
        "Synthesizer Multi-Paper",
        "General Agent (no fake sources)",
    ]

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    for name, passed in zip(names, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    total = sum(results)
    print(f"\n  Total: {total}/{len(results)} test suites passed")
    print(f"  Time:  {elapsed:.1f} seconds")

    if total == len(results):
        print("\n  🎉 ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()