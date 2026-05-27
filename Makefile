.PHONY: all install test lint format clean docker docker-up docker-down run dev

all: install

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --timeout=60

lint:
	ruff check navigator/ tests/

format:
	ruff format navigator/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist *.egg-info

docker:
	docker build -t bastion/navigator:dev .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

run:
	python -m navigator.main --config config/config.yaml

dev:
	python -m navigator.main --config config/config.yaml

.DEFAULT_GOAL := install
