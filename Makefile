.PHONY: setup preprocess train evaluate predict api

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

preprocess:
	python -m src.data_preparation

train:
	python -m src.training

evaluate:
	python -m src.evaluation

predict:
	python -m src.predict_batch

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000
