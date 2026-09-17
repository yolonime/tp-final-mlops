"""Tiny batch-inference script: score a CSV with the serialized model.

Used for the optional Docker / batch-inference deliverable, e.g.:
    python src/churn/predict.py --input data/churn/new.csv --output preds.csv
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd


def main(model_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)

    df["prediction"] = model.predict(df)
    df["probability"] = model.predict_proba(df)[:, 1]
    df.to_csv(output_csv, index=False)
    print(f"Wrote predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/churn/model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    main(args.model, args.input, args.output)
