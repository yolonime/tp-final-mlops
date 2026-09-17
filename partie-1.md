
#  “MLOps Project Template”


Build an MLOps project Template to train the DistilBERT model

---

## What you’ll deliver

**Repo layout**

```
mlops-project/
├─ data/                         # raw & processed (gitignored)
├─ configs/
│  └─ config.yaml                # paths, target, split, model/type/params
├─ src/
│  ├─ config.py                # get the configuration as class
│  ├─ get-data.py                # download the data
│  ├─ preprocess.py              # preprocess the data
│  ├─ pipeline.py                # build ColumnTransformer + model
│  ├─ train.py                   # download the model + train + tune + registry
│  ├─ evaluate.py                # final eval 
│  └─ utils.py                   # small helpers 
├─ tests/
│  └─ test_pipeline.py           # sanity checks for pipeline
├─ Makefile
├─ requirements.txt 
├─ Dockerfile
└─ README.md
```

**Makefile (example)**

```make
PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip -r requirements.txt

train:
	
evaluate:

test:
	pytest -q
```

