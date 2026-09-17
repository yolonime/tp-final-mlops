"""Loading & tokenization helpers for the DistilBERT pipeline."""
from __future__ import annotations

import pandas as pd
from datasets import Dataset


def load_split(csv_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """Load a cached split CSV and normalize column names to text/label."""
    df = pd.read_csv(csv_path)
    return df.rename(columns={text_column: "text", label_column: "label"})


def tokenize_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    """Turn a text/label DataFrame into a tokenized, torch-formatted HF Dataset."""
    dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_length
        )

    dataset = dataset.map(_tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset


def build_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)
