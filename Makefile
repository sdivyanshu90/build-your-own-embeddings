.PHONY: install format lint typecheck test test-unit test-integration test-e2e coverage \
	train-tiny evaluate-tiny build-index-tiny serve docker-build docker-up package smoke

PYTHON ?= python3
MODEL_DIR ?= artifacts/model-tiny
INDEX_DIR ?= artifacts/index-tiny

install:
	$(PYTHON) -m pip install -e ".[dev,faiss]"

format:
	ruff format .

lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy src

test:
	pytest -m "not slow and not network and not gpu" -q

test-unit:
	pytest -m unit -q

test-integration:
	pytest -m integration -q

test-e2e:
	pytest -m end_to_end -q

coverage:
	pytest -m "not slow and not network and not gpu" \
		--cov=embedding_model --cov-branch --cov-report=term-missing

train-tiny:
	embedding-project train --config configs/train_tiny.yaml \
		--data data/sample_pairs.jsonl --output-dir $(MODEL_DIR)

evaluate-tiny:
	embedding-project evaluate --model-path $(MODEL_DIR) \
		--data data/sample_scored_pairs.jsonl \
		--output artifacts/evaluation-tiny.json

build-index-tiny:
	embedding-project index --model-path $(MODEL_DIR) \
		--documents data/sample_documents.jsonl --output-dir $(INDEX_DIR)

serve:
	embedding-project serve --model-path $(MODEL_DIR) --index-path $(INDEX_DIR)

docker-build:
	docker build -f docker/Dockerfile -t embedding-project:local .

docker-up:
	docker compose up --build

package:
	$(PYTHON) -m build

smoke:
	$(PYTHON) scripts/smoke_test.py

