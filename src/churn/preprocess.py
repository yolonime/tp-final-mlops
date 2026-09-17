"""Data loading, target encoding, and train/test split for the churn pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def encode_target(y: pd.Series) -> pd.Series:
    """Encode a binary string target as 0/1.

    The lexicographically greater label (e.g. ">50K" over "<=50K") is treated
    as the positive class, matching `roc_auc_score`'s expectation that 1 is
    the class of interest.
    """
    classes = sorted(y.astype(str).str.strip().unique())
    positive = classes[-1]
    return (y.astype(str).str.strip() == positive).astype(int)


def load_dataset(csv_path: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    y = encode_target(df[target])
    X = df.drop(columns=[target])
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
