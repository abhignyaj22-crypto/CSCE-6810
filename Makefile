.PHONY: install test synthetic smoke full lint
install:
	pip install -e ".[dev,geo,clinical]"
test:
	python -m pytest -q
synthetic:
	python scripts/run_pipeline.py --config configs/synthetic.yaml
smoke:
	python scripts/run_pipeline.py --config configs/smoke.yaml
full:
	python scripts/run_pipeline.py --config configs/default.yaml
lint:
	ruff check src tests scripts
