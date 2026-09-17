"""Build the DistilBERT model for sequence classification."""
from __future__ import annotations


def build_model(model_name: str, num_labels: int):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
