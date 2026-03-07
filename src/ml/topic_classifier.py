"""
Topic Classifier
-----------------
Fine-tunes DistilBERT to classify research papers into topics
based on their title + abstract.

This demonstrates:
- Transfer learning (starting from pre-trained DistilBERT)
- Fine-tuning with HuggingFace Transformers
- Model evaluation with proper train/val/test splits
- Saving and loading trained models

Can train on CPU (your Mac) — takes ~5-10 minutes.
"""

import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


class PaperDataset(Dataset):
    """PyTorch dataset for paper classification."""

    def __init__(self, data: list[dict], tokenizer, label_map: dict, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = self.label_map[item["label"]]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TopicClassifier:
    """
    Fine-tuned DistilBERT for paper topic classification.

    Usage:
        # Training
        classifier = TopicClassifier()
        classifier.train(train_data, val_data, epochs=3)
        classifier.save("models/topic_classifier")

        # Inference
        classifier = TopicClassifier.load("models/topic_classifier")
        result = classifier.predict("Attention Is All You Need. We propose a new...")
        # → {"label": "NLP", "confidence": 0.94, "all_scores": {...}}
    """

    def __init__(self, num_labels: int = 5, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        ).to(self.device)

        # Label mapping (will be set during training or loading)
        self.label_map = {}
        self.reverse_label_map = {}

    def train(
        self,
        train_data: list[dict],
        val_data: list[dict],
        label_map: dict,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ):
        """
        Fine-tune the model on paper classification data.

        Args:
            train_data: List of {"text": ..., "label": ...} dicts
            val_data: Validation data in same format
            label_map: {"NLP": 0, "Computer Vision": 1, ...}
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for AdamW
        """
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}

        # Create datasets
        train_dataset = PaperDataset(train_data, self.tokenizer, label_map)
        val_dataset = PaperDataset(val_data, self.tokenizer, label_map)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        print(f"\nTraining for {epochs} epochs...")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples:   {len(val_data)}")
        print(f"  Batch size:    {batch_size}")
        print(f"  Learning rate: {learning_rate}")

        best_val_acc = 0
        training_log = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Validation
            val_acc, val_report = self._evaluate(val_loader)

            print(f"\n  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

            training_log.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_accuracy": val_acc,
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"\nBest validation accuracy: {best_val_acc:.4f}")

        return training_log

    def _evaluate(self, dataloader) -> tuple:
        """Evaluate model on a dataloader."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=list(self.label_map.keys()),
            output_dict=True,
        )

        return accuracy, report

    def evaluate(self, test_data: list[dict], batch_size: int = 16) -> dict:
        """
        Evaluate on test data and print full classification report.

        Args:
            test_data: Test data in same format as training data

        Returns:
            Dictionary with accuracy and per-class metrics
        """
        test_dataset = PaperDataset(test_data, self.tokenizer, self.label_map)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=list(self.label_map.keys()),
        ))

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def predict(self, text: str) -> dict:
        """
        Classify a single paper's title + abstract.

        Args:
            text: Paper title + abstract

        Returns:
            {"label": "NLP", "confidence": 0.94, "all_scores": {...}}
        """
        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze()

        predicted_idx = torch.argmax(probs).item()
        predicted_label = self.reverse_label_map[predicted_idx]
        confidence = probs[predicted_idx].item()

        all_scores = {
            self.reverse_label_map[i]: round(probs[i].item(), 4)
            for i in range(len(probs))
        }

        return {
            "label": predicted_label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def save(self, path: str):
        """Save the fine-tuned model, tokenizer, and label map."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        with open(os.path.join(path, "label_map.json"), "w") as f:
            json.dump(self.label_map, f)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TopicClassifier":
        """Load a saved model."""
        with open(os.path.join(path, "label_map.json"), "r") as f:
            label_map = json.load(f)

        classifier = cls(num_labels=len(label_map), model_name=path)
        classifier.label_map = label_map
        classifier.reverse_label_map = {v: k for k, v in label_map.items()}

        print(f"Model loaded from {path}")
        return classifier


# ============================================================
# Train and evaluate
# ============================================================
if __name__ == "__main__":
    data_dir = "data/ml"

    # Check if data exists, if not prepare it
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print("Training data not found. Run prepare_classification_data.py first.")
        print("Running data preparation now...")
        from src.ml.prepare_classification_data import prepare_dataset
        prepare_dataset(papers_per_category=100)

    # Load data
    print("\nLoading data...")
    with open(os.path.join(data_dir, "train.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, "val.json")) as f:
        val_data = json.load(f)
    with open(os.path.join(data_dir, "test.json")) as f:
        test_data = json.load(f)
    with open(os.path.join(data_dir, "label_map.json")) as f:
        label_map = json.load(f)

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize and train
    classifier = TopicClassifier(num_labels=len(label_map))

    training_log = classifier.train(
        train_data=train_data,
        val_data=val_data,
        label_map=label_map,
        epochs=10,
        batch_size=8,
        learning_rate=5e-5,
    )

    # Evaluate on test set
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    results = classifier.evaluate(test_data)

    # Save the model
    model_path = "models/topic_classifier"
    classifier.save(model_path)

    # Test with a sample prediction
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    test_texts = [
        "Attention Is All You Need. We propose a new network architecture based on attention mechanisms, dispensing with recurrence and convolutions entirely.",
        "ImageNet Classification with Deep Convolutional Neural Networks. We trained a large deep convolutional neural network to classify images in the ImageNet dataset.",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. We explore retrieval-augmented generation models for knowledge-intensive NLP tasks.",
    ]

    for text in test_texts:
        result = classifier.predict(text)
        print(f"\n  Text: {text[:80]}...")
        print(f"  → {result['label']} (confidence: {result['confidence']:.2%})")
        print(f"    Scores: {result['all_scores']}")

    # Save training log
    with open(os.path.join(data_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n✅ Training complete!")