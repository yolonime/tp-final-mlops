"""Fine-tune DistilBERT for SST-2 sentiment classification with MLflow tracking.

Usage:
    python src/distilbert/train.py --config configs/distilbert.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/distilbert/train.py` to resolve `src.*` imports without
# requiring PYTHONPATH to be set manually.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.common.mlflow_utils import setup_mlflow
from src.distilbert.config import get_config
from src.distilbert.pipeline import build_model
from src.distilbert.preprocess import build_tokenizer, load_split, tokenize_dataset
from src.distilbert.utils import compute_metrics


def main(config_path: str) -> None:
    cfg = get_config(config_path)
    setup_mlflow(cfg.mlflow.experiment_name)

    tokenizer = build_tokenizer(cfg.model.name)
    train_df = load_split("data/distilbert/train.csv", cfg.data.text_column, cfg.data.label_column)
    eval_df = load_split("data/distilbert/validation.csv", cfg.data.text_column, cfg.data.label_column)
    train_ds = tokenize_dataset(train_df, tokenizer, cfg.data.max_length)
    eval_ds = tokenize_dataset(eval_df, tokenizer, cfg.data.max_length)

    model = build_model(cfg.model.name, cfg.model.num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        seed=cfg.train.seed,
        report_to=["mlflow"],
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run(run_name=f"{cfg.model.name}-finetune"):
        trainer.train()
        metrics = trainer.evaluate()
        print(metrics)

        trainer.save_model(cfg.train.output_dir)
        tokenizer.save_pretrained(cfg.train.output_dir)

        registered_name = getattr(cfg.mlflow, "registered_model_name", None)
        if registered_name:
            try:
                from mlflow import transformers as mlflow_transformers

                mlflow_transformers.log_model(
                    transformers_model={"model": trainer.model, "tokenizer": tokenizer},
                    artifact_path="model",
                    registered_model_name=registered_name,
                )
            except Exception as exc:  # noqa: BLE001 - optional registry step must not fail the run
                print(f"Skipping MLflow model registration ({exc}); "
                      f"the fine-tuned model is still saved at {cfg.train.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/distilbert.yaml")
    args = parser.parse_args()
    main(args.config)
