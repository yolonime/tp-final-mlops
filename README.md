# MLOps Final Project

This repo implements the two exercises described in [`proposal.md`](proposal.md)
and [`partie-1.md`](partie-1.md) as a single project with two independent,
parallel training pipelines that share the same conventions (config-driven,
Makefile-orchestrated, MLflow-tracked):

| Pipeline | Spec | Task | Dataset | Code |
|---|---|---|---|---|
| **Churn classifier** | `proposal.md` | Tabular binary classification (scikit-learn `Pipeline` + `ColumnTransformer`) | [Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) (>=50K vs <50K) | [`src/churn/`](src/churn) |
| **DistilBERT sentiment** | `partie-1.md` | NLP fine-tuning (HuggingFace `transformers`) | [SST-2](https://huggingface.co/datasets/glue) (GLUE) sentiment | [`src/distilbert/`](src/distilbert) |

Both pipelines log params/metrics/artifacts to **MLflow** and (optionally)
register their best model to the MLflow Model Registry.

## Live demo

**[Census Income Estimator](docs/index.html)** — a static page that
reproduces the trained churn classifier's exact predictions (same
coefficients, standardization, and one-hot encoding as `artifacts/churn/model.joblib`)
entirely client-side in JavaScript. Fill in the form or step through real
held-out records to see the model's probability and a full breakdown of
every feature's contribution to the logistic regression score.

Enable **GitHub Pages** (Settings → Pages → Source: `Deploy from branch`,
branch `master`, folder `/docs`) to serve it at
`https://yolonime.github.io/tp-final-mlops/`, or just open
[`docs/index.html`](docs/index.html) directly in a browser — no server needed.

## Repo layout

```
MLOPSFINAL/
├─ data/                         # raw & processed (gitignored)
│  ├─ churn/
│  └─ distilbert/
├─ configs/
│  ├─ churn.yaml                 # Adult Income pipeline config
│  └─ distilbert.yaml            # SST-2 / DistilBERT pipeline config
├─ src/
│  ├─ common/                    # shared config loader + MLflow setup
│  ├─ churn/                     # tabular pipeline (get_data/preprocess/pipeline/train/evaluate/predict)
│  └─ distilbert/                # NLP pipeline (get_data/preprocess/pipeline/train/evaluate)
├─ tests/
│  ├─ test_churn_pipeline.py
│  └─ test_distilbert_pipeline.py
├─ docs/
│  └─ index.html                 # live demo (see "Live demo" below)
├─ Makefile
├─ requirements.txt
├─ pytest.ini
├─ .env.example
├─ Dockerfile
└─ README.md
```

## Setup

```bash
make init                       # creates .venv and installs requirements.txt
cp .env.example .env            # then edit as needed (MLflow tracking URI, etc.)
```

## Churn classifier (Adult Income)

```bash
make get-data-churn             # downloads data/churn/adult.csv from UCI
make train-churn                # GridSearchCV + mlflow.autolog + optional registry
make evaluate-churn             # ROC/PR/confusion-matrix plots + predictions.csv logged to MLflow
```

Config: [`configs/churn.yaml`](configs/churn.yaml) (dataset path, feature
lists, model type/params, CV strategy).

## DistilBERT sentiment (SST-2)

```bash
make get-data-distilbert        # caches train.csv / validation.csv via HuggingFace `datasets`
make train-distilbert           # fine-tunes distilbert-base-uncased, reports to MLflow
make evaluate-distilbert        # accuracy/F1/ROC-AUC + confusion matrix logged to MLflow
```

Config: [`configs/distilbert.yaml`](configs/distilbert.yaml) (model name,
tokenization, training hyperparameters). Note: SST-2's official `test` split
is unlabeled, so training uses `train` and final evaluation uses the official
`validation` split.

> The DistilBERT pipeline downloads ~260MB of pretrained weights on first run
> and trains meaningfully faster on a GPU; on CPU-only machines, lower
> `train.num_train_epochs` in the config for a quick smoke test.

## MLflow UI

```bash
make mlflow-ui                  # serves http://localhost:5000 from sqlite:///mlflow.db
```

> Tracking backend: `sqlite:///mlflow.db` (MLflow 3.x deprecated the plain
> filesystem store and requires a database backend for full Model Registry
> support). Override via `MLFLOW_TRACKING_URI` in `.env` if you'd rather
> point at a remote MLflow server.

## Tests & linting

```bash
make test                       # offline unit tests only (pytest -m "not network")
make test-all                   # includes the DistilBERT tokenizer test (downloads weights)
make lint                       # ruff check src tests
```

## Docker

```bash
make docker-build
make docker-run                 # runs the churn training job by default; override CMD for other scripts
```
