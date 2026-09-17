"""Final evaluation of a fine-tuned DistilBERT checkpoint: metrics + confusion
matrix logged as MLflow artifacts.

Usage:
    python src/distilbert/evaluate.py --config configs/distilbert.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python src/distilbert/evaluate.py` to resolve `src.*` imports without
# requiring PYTHONPATH to be set manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.common.mlflow_utils import setup_mlflow
from src.distilbert.config import get_config
from src.distilbert.preprocess import load_split, tokenize_dataset
from src.distilbert.utils import compute_metrics, plot_confusion_matrix


def main(config_path: str, model_dir: str) -> None:
    cfg = get_config(config_path)
    setup_mlflow(cfg.mlflow.experiment_name)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    eval_df = load_split("data/distilbert/validation.csv", cfg.data.text_column, cfg.data.label_column)
    eval_ds = tokenize_dataset(eval_df, tokenizer, cfg.data.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(eval_ds, batch_size=32, collate_fn=collator)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            outputs = model(**batch)
            all_logits.append(outputs.logits.numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics((logits, labels))
    print(metrics)

    preds = np.argmax(logits, axis=-1)
    os.makedirs("artifacts/distilbert", exist_ok=True)
    cm_path = "artifacts/distilbert/confusion_matrix.png"
    plot_confusion_matrix(labels, preds, cm_path)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distilbert.yaml")
    parser.add_argument("--model-dir", default="artifacts/distilbert")
    args = parser.parse_args()
    main(args.config, args.model_dir)
