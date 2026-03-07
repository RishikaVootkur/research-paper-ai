"""
Classification Data Preparation
--------------------------------
Fetches papers from ArXiv with their categories and creates
a training dataset for the topic classifier.

ArXiv categories we'll classify:
- cs.CL  → NLP (Natural Language Processing)
- cs.CV  → Computer Vision
- cs.LG  → Machine Learning
- cs.AI  → Artificial Intelligence
- cs.IR  → Information Retrieval

We fetch papers from each category to build a balanced dataset.
"""

import os
import sys
import json
import random
import arxiv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# Map ArXiv categories to human-readable labels
CATEGORY_MAP = {
    "cs.CL": "NLP",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval",
}

CATEGORIES = list(CATEGORY_MAP.keys())
LABELS = list(CATEGORY_MAP.values())


def fetch_papers_by_category(category: str, max_results: int = 100) -> list[dict]:
    """
    Fetch papers from a specific ArXiv category.

    Args:
        category: ArXiv category code (e.g., "cs.CL")
        max_results: How many papers to fetch

    Returns:
        List of {text, label} dicts
    """
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    client = arxiv.Client()
    papers = []

    for result in client.results(search):
        # Use title + abstract as the input text
        # This gives the classifier enough context to categorize
        text = f"{result.title}. {result.summary}"
        text = text.replace("\n", " ").strip()

        papers.append({
            "text": text,
            "label": CATEGORY_MAP[category],
            "category": category,
            "title": result.title.replace("\n", " "),
        })

    return papers


def prepare_dataset(papers_per_category: int = 100, output_dir: str = "data/ml"):
    """
    Build the full classification dataset.

    Fetches papers from each category and splits into train/val/test.

    Args:
        papers_per_category: How many papers to fetch per category
        output_dir: Where to save the dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    all_papers = []

    print(f"Fetching {papers_per_category} papers per category...")
    for category in CATEGORIES:
        label = CATEGORY_MAP[category]
        print(f"\n  Fetching {category} ({label})...")
        papers = fetch_papers_by_category(category, max_results=papers_per_category)
        print(f"    Got {len(papers)} papers")
        all_papers.extend(papers)

    # Shuffle
    random.seed(42)
    random.shuffle(all_papers)

    # Split: 70% train, 15% val, 15% test
    total = len(all_papers)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)

    train_data = all_papers[:train_end]
    val_data = all_papers[train_end:val_end]
    test_data = all_papers[val_end:]

    # Save
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        filepath = os.path.join(output_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved {name}: {len(data)} samples → {filepath}")

    # Print distribution
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total: {total} papers")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"\nLabel distribution (train):")
    from collections import Counter
    label_counts = Counter(p["label"] for p in train_data)
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    # Save label mapping
    label_map = {label: i for i, label in enumerate(LABELS)}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabel map: {label_map}")

    return train_data, val_data, test_data


if __name__ == "__main__":
    prepare_dataset(papers_per_category=100)