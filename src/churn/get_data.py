"""Download the Adult Income (Census Income) dataset from the UCI ML repository.

Usage:
    python src/churn/get_data.py --output data/churn/adult.csv
"""
from __future__ import annotations

import argparse
import os
import urllib.request

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]


def download(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    raw_path = output_path + ".raw"

    print(f"Downloading Adult Income dataset from {UCI_URL} ...")
    urllib.request.urlretrieve(UCI_URL, raw_path)

    df = pd.read_csv(raw_path, header=None, names=COLUMNS, skipinitialspace=True)
    df = df.dropna()
    # Labels come as " <=50K" / " >50K." (train) -- normalize to "<=50K" / ">50K".
    df["income"] = df["income"].str.strip().str.rstrip(".")

    df.to_csv(output_path, index=False)
    os.remove(raw_path)
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/churn/adult.csv")
    args = parser.parse_args()
    download(args.output)
