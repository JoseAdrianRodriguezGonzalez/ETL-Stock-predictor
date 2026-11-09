.PHONY: lint fix format check test

lint:
	uv run --project collector ruff check collector/

fix:
	uv run --project collector ruff check collector/ --fix

format:
	uv run --project collector ruff format collector/

check:
	uv run --project collector ruff check collector/
	uv run --project collector ruff format --check collector/

test:
	uv run --project collector pytest -v
