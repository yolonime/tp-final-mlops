"""Download the SST-2 sentiment dataset (GLUE) via HuggingFace `datasets` and
cache it locally as CSVs.

Usage:
    python src/distilbert/get_data.py --output data/distilbert
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset


def download(hf_dataset: list[str], text_column: str, label_column: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset(*hf_dataset)

    # GLUE's SST-2 "test" split has no public labels, so we only cache the
    # labeled splits: train (for fine-tuning) and validation (held out for
    # evaluation, see configs/distilbert.yaml).
    for split in ("train", "validation"):
        df = ds[split].to_pandas()[[text_column, label_column]]
        out_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=["glue", "sst2"])
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output", default="data/distilbert")
    args = parser.parse_args()
    download(args.dataset, args.text_column, args.label_column, args.output)
