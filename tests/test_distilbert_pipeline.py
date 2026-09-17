"""Sanity checks for the DistilBERT sentiment pipeline.

The metrics/data-shaping tests run offline. The tokenizer/model test needs
network access to download pretrained weights on first run, so it's tagged
`network` and skipped by `make test` (see pytest.ini / Makefile); run
`make test-all` (or `pytest -m network`) to include it.
"""
import numpy as np
import pandas as pd
import pytest

from src.distilbert.preprocess import load_split
from src.distilbert.utils import compute_metrics


def test_compute_metrics_perfect_predictions():
    logits = np.array([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]])
    labels = np.array([0, 1, 0, 1])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_load_split_renames_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"sentence": ["good movie", "bad movie"], "label": [1, 0]}).to_csv(
        csv_path, index=False
    )
    df = load_split(str(csv_path), text_column="sentence", label_column="label")
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 2


@pytest.mark.network
def test_tokenize_dataset_produces_tensors():
    from src.distilbert.preprocess import build_tokenizer, tokenize_dataset

    tokenizer = build_tokenizer("distilbert-base-uncased")
    df = pd.DataFrame({"text": ["good movie", "bad movie"], "label": [1, 0]})
    dataset = tokenize_dataset(df, tokenizer, max_length=16)

    assert "input_ids" in dataset.column_names
    assert len(dataset) == 2
