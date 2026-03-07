"""
RAG Evaluation
--------------
Evaluates the RAG pipeline using RAGAS-inspired metrics.

Since the full RAGAS library requires OpenAI API keys (paid),
we implement lightweight evaluation metrics that measure the
same concepts using our free Groq LLM.

Metrics:
1. Faithfulness:      Does the answer stay grounded in sources?
2. Answer Relevancy:  Does the answer address the question?
3. Context Relevancy:  Are retrieved chunks relevant to the question?
4. Citation Accuracy:  Does the answer cite sources properly?
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.rag.rag_chain import RAGChain

load_dotenv()


# ============================================================
# Evaluation dataset
# ============================================================

EVAL_DATASET = [
    {
        "question": "What is LoRA and how does it work?",
        "expected_topics": ["low-rank", "adaptation", "fine-tuning", "trainable parameters"],
        "expected_papers": ["LoRA"],
    },
    {
        "question": "How does the attention mechanism work in transformers?",
        "expected_topics": ["query", "key", "value", "multi-head", "self-attention"],
        "expected_papers": ["Attention"],
    },
    {
        "question": "What is retrieval augmented generation?",
        "expected_topics": ["retrieval", "generation", "knowledge", "non-parametric"],
        "expected_papers": ["Retrieval-Augmented"],
    },
    {
        "question": "What metrics are used to evaluate RAG systems?",
        "expected_topics": ["precision", "recall", "faithfulness", "BLEU", "ROUGE"],
        "expected_papers": ["RAG", "Retrieval"],
    },
    {
        "question": "How does LoRA reduce GPU memory requirements?",
        "expected_topics": ["low-rank", "parameters", "memory", "decomposition"],
        "expected_papers": ["LoRA"],
    },
    {
        "question": "What is the transformer architecture?",
        "expected_topics": ["encoder", "decoder", "attention", "layers"],
        "expected_papers": ["Attention"],
    },
    {
        "question": "How can RAG systems handle noisy retrieved documents?",
        "expected_topics": ["noise", "retrieval", "quality", "filtering"],
        "expected_papers": ["RAG", "Retrieval", "Engineering"],
    },
    {
        "question": "What is the difference between LoRA and full fine-tuning?",
        "expected_topics": ["parameters", "memory", "performance", "rank"],
        "expected_papers": ["LoRA"],
    },
    {
        "question": "What are vector databases used for in RAG?",
        "expected_topics": ["embedding", "similarity", "retrieval", "search"],
        "expected_papers": ["RAG", "Retrieval", "HAKES"],
    },
    {
        "question": "What is the current stock price of Apple?",
        "expected_topics": [],
        "expected_papers": [],
        "is_out_of_scope": True,
    },
]


# ============================================================
# LLM-based evaluators
# ============================================================

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You evaluate whether an answer is faithful to the provided sources.
Rate from 1-5:
5 = Every claim is directly supported by the sources
4 = Most claims supported, minor unsupported details
3 = Some claims supported, some not verifiable from sources
2 = Many unsupported claims
1 = Mostly hallucinated or fabricated

Respond with ONLY a JSON: {{"score": <1-5>, "reason": "<brief explanation>"}}"""),
    ("human", "SOURCES:\n{context}\n\nANSWER:\n{answer}\n\nRate faithfulness:"),
])

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You evaluate whether an answer addresses the question asked.
Rate from 1-5:
5 = Directly and completely answers the question
4 = Mostly answers the question with minor gaps
3 = Partially answers the question
2 = Barely addresses the question
1 = Does not answer the question at all

Respond with ONLY a JSON: {{"score": <1-5>, "reason": "<brief explanation>"}}"""),
    ("human", "QUESTION: {question}\n\nANSWER:\n{answer}\n\nRate relevancy:"),
])


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-based metrics.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=200,
        )
        self.parser = StrOutputParser()

        self.faithfulness_chain = FAITHFULNESS_PROMPT | self.llm | self.parser
        self.relevancy_chain = RELEVANCY_PROMPT | self.llm | self.parser

    def evaluate_single(self, rag: RAGChain, test_case: dict) -> dict:
        """Evaluate a single question."""
        question = test_case["question"]
        is_oos = test_case.get("is_out_of_scope", False)

        # Get RAG response
        result = rag.query_with_details(question)
        answer = result["answer"]
        context = result.get("formatted_context", "")
        sources = result.get("sources", [])

        scores = {}

        # 1. Topic Coverage: Do expected keywords appear in the answer?
        if test_case["expected_topics"]:
            answer_lower = answer.lower()
            found = sum(1 for t in test_case["expected_topics"] if t.lower() in answer_lower)
            scores["topic_coverage"] = round(found / len(test_case["expected_topics"]), 2)
        else:
            scores["topic_coverage"] = None

        # 2. Source Accuracy: Did we retrieve from expected papers?
        if test_case["expected_papers"]:
            source_titles = " ".join(s.get("title", "") for s in sources).lower()
            found = sum(1 for p in test_case["expected_papers"] if p.lower() in source_titles)
            scores["source_accuracy"] = round(found / len(test_case["expected_papers"]), 2)
        else:
            scores["source_accuracy"] = None

        # 3. Citation Presence: Does the answer have citations?
        has_citations = any(marker in answer for marker in ["[1]", "[2]", "[Source", "References:"])
        scores["has_citations"] = has_citations

        # 4. Out-of-scope handling
        if is_oos:
            oos_phrases = ["cannot", "no information", "not available",
                          "don't have", "not found", "do not contain",
                          "outside", "no relevant"]
            scores["handles_oos"] = any(p in answer.lower() for p in oos_phrases)

        # 5. LLM-based faithfulness (if we have context)
        if context and not is_oos:
            try:
                resp = self.faithfulness_chain.invoke({"context": context[:3000], "answer": answer[:2000]})
                clean = resp.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                scores["faithfulness"] = parsed.get("score", 3)
            except Exception:
                scores["faithfulness"] = None

        # 6. LLM-based relevancy
        if not is_oos:
            try:
                resp = self.relevancy_chain.invoke({"question": question, "answer": answer[:2000]})
                clean = resp.strip().replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                scores["relevancy"] = parsed.get("score", 3)
            except Exception:
                scores["relevancy"] = None

        return {
            "question": question,
            "scores": scores,
            "num_sources": len(sources),
            "answer_length": len(answer),
        }

    def evaluate_all(self, rag: RAGChain, dataset: list[dict] = None) -> dict:
        """Run full evaluation across the dataset."""
        if dataset is None:
            dataset = EVAL_DATASET

        print(f"\n{'='*60}")
        print(f"  RAG EVALUATION ({len(dataset)} questions)")
        print(f"{'='*60}")

        results = []
        for i, test_case in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] {test_case['question'][:50]}...")
            result = self.evaluate_single(rag, test_case)
            results.append(result)

            scores = result["scores"]
            for metric, value in scores.items():
                if value is not None:
                    print(f"   {metric}: {value}")

            # Small delay to avoid rate limits
            time.sleep(1)

        # Compute aggregates
        summary = self._compute_summary(results)
        self._print_summary(summary)

        # Save results
        output = {"results": results, "summary": summary}
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to data/processed/evaluation_results.json")

        return output

    def _compute_summary(self, results: list[dict]) -> dict:
        """Compute aggregate metrics."""
        summary = {}

        # Average each numeric metric
        for metric in ["topic_coverage", "source_accuracy", "faithfulness", "relevancy"]:
            values = [r["scores"].get(metric) for r in results if r["scores"].get(metric) is not None]
            if values:
                summary[metric] = {
                    "mean": round(sum(values) / len(values), 2),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Citation rate
        citation_values = [r["scores"].get("has_citations") for r in results
                         if r["scores"].get("has_citations") is not None]
        if citation_values:
            summary["citation_rate"] = round(sum(citation_values) / len(citation_values), 2)

        # OOS handling
        oos_values = [r["scores"].get("handles_oos") for r in results
                     if r["scores"].get("handles_oos") is not None]
        if oos_values:
            summary["oos_handling_rate"] = round(sum(oos_values) / len(oos_values), 2)

        return summary

    def _print_summary(self, summary: dict):
        """Pretty-print the evaluation summary."""
        print(f"\n{'='*60}")
        print("  EVALUATION SUMMARY")
        print(f"{'='*60}")

        for metric, data in summary.items():
            if isinstance(data, dict):
                print(f"\n  {metric}:")
                print(f"    Mean: {data['mean']}")
                print(f"    Range: [{data['min']} — {data['max']}]")
                print(f"    Samples: {data['count']}")
            else:
                print(f"\n  {metric}: {data}")


# ============================================================
# Run evaluation
# ============================================================
if __name__ == "__main__":
    print("Initializing RAG chain...")
    rag = RAGChain(use_hyde="auto")

    evaluator = RAGEvaluator()
    evaluator.evaluate_all(rag)