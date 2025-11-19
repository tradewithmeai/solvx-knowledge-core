.PHONY: help install dev test fmt lint clean migrate upgrade downgrade

help:
	@echo "SolVX Knowledge Core - Development Commands"
	@echo ""
	@echo "install    - Install dependencies with poetry"
	@echo "dev        - Run backend in development mode"
	@echo "test       - Run tests with coverage"
	@echo "fmt        - Format code with black and isort"
	@echo "lint       - Lint code with ruff and mypy"
	@echo "clean      - Remove build artifacts and cache"
	@echo "migrate    - Create new Alembic migration"
	@echo "upgrade    - Run Alembic migrations"
	@echo "downgrade  - Rollback one Alembic migration"

install:
	poetry install --no-root

dev:
	poetry run uvicorn backend.app.main:app --host 127.0.0.1 --port 8787 --reload

test:
	poetry run pytest

fmt:
	poetry run black backend cli tests
	poetry run isort backend cli tests

lint:
	poetry run ruff backend cli tests
	poetry run mypy backend cli

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov .coverage build dist *.egg-info

migrate:
	@read -p "Migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

upgrade:
	poetry run alembic upgrade head

downgrade:
	poetry run alembic downgrade -1
