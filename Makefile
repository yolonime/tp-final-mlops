PY=python
ENV?=.venv
EXP_CHURN?=adult-income-churn-exp
EXP_DISTILBERT?=distilbert-sst2-exp

.PHONY: init \
        get-data-churn train-churn evaluate-churn \
        get-data-distilbert train-distilbert evaluate-distilbert \
        test test-all lint mlflow-ui docker-build docker-run

init:
	$(PY) -m venv $(ENV) && . $(ENV)/bin/activate && pip install -U pip -r requirements.txt

## --- Churn classifier (tabular, sklearn + MLflow) ---

get-data-churn:
	$(PY) src/churn/get_data.py --output data/churn/adult.csv

train-churn:
	MLFLOW_EXPERIMENT_NAME=$(EXP_CHURN) $(PY) src/churn/train.py --config configs/churn.yaml

evaluate-churn:
	MLFLOW_EXPERIMENT_NAME=$(EXP_CHURN) $(PY) src/churn/evaluate.py --config configs/churn.yaml

## --- DistilBERT sentiment classifier (NLP, HuggingFace + MLflow) ---

get-data-distilbert:
	$(PY) src/distilbert/get_data.py --output data/distilbert

train-distilbert:
	MLFLOW_EXPERIMENT_NAME=$(EXP_DISTILBERT) $(PY) src/distilbert/train.py --config configs/distilbert.yaml

evaluate-distilbert:
	MLFLOW_EXPERIMENT_NAME=$(EXP_DISTILBERT) $(PY) src/distilbert/evaluate.py --config configs/distilbert.yaml

## --- Common ---

test:
	pytest -q -m "not network"

test-all:
	pytest -q

lint:
	ruff check src tests

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

docker-build:
	docker build -t mlops-final .

docker-run:
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/mlruns:/app/mlruns mlops-final
