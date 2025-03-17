# OpenAI Agents Python SDK Guide

## Build Commands
- Install dependencies: `uv sync --all-extras --all-packages --group dev`
- Run tests: `uv run pytest` or `make tests`
- Run single test: `uv run pytest tests/path_to_test.py::test_name -v`
- Lint code: `uv run ruff check` or `make lint`
- Format code: `uv run ruff format` or `make format`
- Type check: `uv run mypy .` or `make mypy`
- Build docs: `uv run mkdocs build` or `make build-docs`

## Code Style Guidelines
- Line length: 100 characters max
- Use Google docstring convention
- Type hints: Project aims for strict typing (mypy)
- Imports: Use isort with combine-as-imports, known-first-party=["agents"]
- Error handling: Use typed exceptions from src/agents/exceptions.py
- Naming: Use snake_case for variables/functions, PascalCase for classes
- Testing: Use pytest with asyncio mode="auto"
- Use Pydantic >=2.10 for data validation
- Format with ruff (version 0.9.2)