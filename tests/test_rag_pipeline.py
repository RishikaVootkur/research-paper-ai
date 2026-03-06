"""
RAG Pipeline Tests
------------------
Tests the full RAG pipeline with diverse question types.
Checks that the system retrieves relevant content and generates
grounded answers.

Run: python tests/test_rag_pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag.rag_chain import RAGChain
from src.rag.prompts import classify_question


def test_question_classifier():
    """Test that the question classifier labels correctly."""
    print("\n📋 TEST: Question Classifier")
    print("-" * 40)

    test_cases = [
        ("What is LoRA?", "factual"),
        ("How does attention work in transformers?", "methodology"),
        ("Compare LoRA and full fine-tuning", "comparison"),
        ("What are the main trends in RAG research?", "summary"),
    ]

    passed = 0
    for question, expected_type in test_cases:
        actual = classify_question(question)
        status = "✅" if actual == expected_type else "❌"
        if actual == expected_type:
            passed += 1
        print(f"  {status} '{question[:45]}...' → {actual} (expected: {expected_type})")

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_retrieval_relevance(rag: RAGChain):
    """Test that retrieval returns relevant papers for known queries."""
    print("\n🔍 TEST: Retrieval Relevance")
    print("-" * 40)

    # Each test case: (query, expected_paper_substring)
    # We check that at least one retrieved source contains the expected paper
    test_cases = [
        ("How does LoRA work?", "LoRA"),
        ("Explain the transformer attention mechanism", "Attention"),
        ("What metrics does RAGAS define?", "Ragas"),
        ("How does retrieval augmented generation work?", "Retrieval-Augmented"),
    ]

    passed = 0
    for query, expected_paper in test_cases:
        result = rag.query(query)
        source_titles = [s["title"] for s in result["sources"]]
        found = any(expected_paper.lower() in t.lower() for t in source_titles)
        status = "✅" if found else "❌"
        if found:
            passed += 1
        print(f"  {status} '{query[:40]}...'")
        print(f"     Expected paper containing: '{expected_paper}'")
        print(f"     Got: {[t[:50] for t in source_titles]}")

    print(f"\n  Result: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_answer_has_citations(rag: RAGChain):
    """Test that answers contain citations."""
    print("\n📝 TEST: Citation Presence")
    print("-" * 40)

    questions = [
        "What is LoRA?",
        "How does RAG reduce hallucination?",
        "Explain multi-head attention",
    ]

    passed = 0
    for question in questions:
        result = rag.query(question)
        answer = result["answer"]

        # Check for citation markers [1], [2], etc. or [Source
        has_citations = (
            "[1]" in answer or
            "[Source" in answer or
            "References:" in answer or
            "[\"" in answer
        )

        status = "✅" if has_citations else "❌"
        if has_citations:
            passed += 1
        print(f"  {status} '{question[:40]}...' — citations found: {has_citations}")

    print(f"\n  Result: {passed}/{len(questions)} passed")
    return passed == len(questions)


def test_grounding(rag: RAGChain):
    """Test that the system admits when it doesn't have relevant info."""
    print("\n🛡️ TEST: Grounding (handles unknown topics)")
    print("-" * 40)

    # Ask about something definitely NOT in our papers
    question = "What is the current stock price of Apple Inc?"
    result = rag.query(question)
    answer = result["answer"].lower()

    # The answer should indicate it can't answer from sources
    admits_limitation = any(phrase in answer for phrase in [
        "cannot", "don't have", "not found",
        "no relevant", "sources do not",
        "not available", "unable to",
        "based on the available sources",
        "no information available",
        "do not contain", "does not contain",
        "not mentioned", "no mention",
        "outside the scope",
    ])

    status = "✅" if admits_limitation else "❌"
    print(f"  {status} Out-of-domain question — admits limitation: {admits_limitation}")
    print(f"     Answer preview: {result['answer'][:200]}...")

    print(f"\n  Result: {'1/1 passed' if admits_limitation else '0/1 passed'}")
    return admits_limitation


def test_multi_paper_synthesis(rag: RAGChain):
    """Test that answers draw from multiple papers when appropriate."""
    print("\n📚 TEST: Multi-Paper Synthesis")
    print("-" * 40)

    question = "What techniques exist for efficient fine-tuning of large language models?"
    result = rag.query(question)

    num_papers = result["num_papers"]
    status = "✅" if num_papers >= 2 else "❌"
    print(f"  {status} Retrieved from {num_papers} papers (expected >= 2)")
    print(f"     Sources: {[s['title'][:40] for s in result['sources']]}")

    print(f"\n  Result: {'1/1 passed' if num_papers >= 2 else '0/1 passed'}")
    return num_papers >= 2


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 60)
    print("  RAG PIPELINE TEST SUITE")
    print("=" * 60)

    start_time = time.time()

    # Test 1: Classifier (no RAG chain needed)
    t1 = test_question_classifier()

    # Initialize RAG chain once for all remaining tests
    print("\n⏳ Initializing RAG chain...")
    rag = RAGChain(use_hyde="auto")

    # Test 2-5: Need RAG chain
    t2 = test_retrieval_relevance(rag)
    t3 = test_answer_has_citations(rag)
    t4 = test_grounding(rag)
    t5 = test_multi_paper_synthesis(rag)

    elapsed = time.time() - start_time

    # Summary
    results = [t1, t2, t3, t4, t5]
    test_names = [
        "Question Classifier",
        "Retrieval Relevance",
        "Citation Presence",
        "Grounding (unknown topics)",
        "Multi-Paper Synthesis",
    ]

    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    for name, passed in zip(test_names, results):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    total_passed = sum(results)
    print(f"\n  Total: {total_passed}/{len(results)} test suites passed")
    print(f"  Time:  {elapsed:.1f} seconds")

    if total_passed == len(results):
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {len(results) - total_passed} test suite(s) failed")

    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()