"""
Paper Classification Integration
---------------------------------
Classifies all papers in the database by topic using the
fine-tuned DistilBERT model. Adds topic labels to paper metadata.

Also provides a utility to classify new papers during ingestion.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.ml.topic_classifier import TopicClassifier


def classify_all_papers(
    metadata_path: str = "data/processed/ingestion_log.json",
    model_path: str = "models/topic_classifier",
    output_path: str = "data/processed/paper_topics.json",
):
    """
    Classify all ingested papers and save topic labels.
    """
    # Load classifier
    print("Loading topic classifier...")
    classifier = TopicClassifier.load(model_path)

    # Load paper metadata
    with open(metadata_path, "r") as f:
        log = json.load(f)

    papers = log.get("ingested_papers", {})
    print(f"Classifying {len(papers)} papers...")

    results = {}
    for paper_id, info in papers.items():
        title = info.get("title", "")
        # Use title as input (we don't have abstracts stored, but title works)
        prediction = classifier.predict(title)

        results[paper_id] = {
            "title": title,
            "predicted_topic": prediction["label"],
            "confidence": prediction["confidence"],
            "all_scores": prediction["all_scores"],
        }

        conf_pct = prediction["confidence"] * 100
        print(f"  [{prediction['label']:25s}] ({conf_pct:5.1f}%) {title[:55]}...")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved topic classifications to {output_path}")

    # Print summary
    from collections import Counter
    topic_counts = Counter(r["predicted_topic"] for r in results.values())
    print(f"\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic}: {count} papers")

    return results


if __name__ == "__main__":
    classify_all_papers()