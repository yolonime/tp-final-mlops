FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV PYTHONUNBUFFERED=1

# Default: train the (lighter) churn classifier. Override the command to run
# the DistilBERT pipeline or the batch-inference script instead, e.g.:
#   docker run --rm mlops-final python src/churn/predict.py --input data/churn/new.csv
CMD ["python", "src/churn/train.py", "--config", "configs/churn.yaml"]
