# Genetic MCP Makefile

.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: install
install: ## Install the package in development mode
	uv pip install -e .

.PHONY: install-dev
install-dev: ## Install development dependencies
	uv pip install -e ".[dev]"

.PHONY: test
test: ## Run all tests
	uv run pytest -v tests/

.PHONY: test-unit
test-unit: ## Run unit tests only
	uv run pytest -v tests/ -k "not integration"

.PHONY: test-integration
test-integration: ## Run integration tests only
	uv run pytest -v tests/integration/

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	uv run pytest --cov=genetic_mcp --cov-report=html --cov-report=term-missing tests/

.PHONY: lint
lint: ## Run all linting checks (ruff, mypy)
	@echo "Running ruff..."
	uv run ruff check genetic_mcp/ tests/
	@echo "\nRunning mypy..."
	uv run mypy genetic_mcp/ tests/

.PHONY: lint-fix
lint-fix: ## Run ruff with auto-fix
	uv run ruff check --fix genetic_mcp/ tests/

.PHONY: format
format: ## Format code with ruff
	uv run ruff format genetic_mcp/ tests/

.PHONY: type-check
type-check: ## Run mypy type checking only
	uv run mypy genetic_mcp/ tests/

.PHONY: clean
clean: ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

.PHONY: build
build: clean ## Build distribution packages
	uv build

.PHONY: run
run: ## Run the Genetic MCP server
	genetic-mcp

.PHONY: run-stdio
run-stdio: ## Run server in stdio mode
	GENETIC_MCP_TRANSPORT=stdio genetic-mcp

.PHONY: run-http
run-http: ## Run server in HTTP mode with SSE
	GENETIC_MCP_TRANSPORT=http genetic-mcp

.PHONY: debug
debug: ## Run server with debug logging
	GENETIC_MCP_DEBUG=true genetic-mcp

.PHONY: check
check: lint test ## Run all checks (lint + test)

.PHONY: pre-commit
pre-commit: format lint test-unit ## Run pre-commit checks

.PHONY: docs
docs: ## Generate documentation (placeholder for future)
	@echo "Documentation generation not yet implemented"

.PHONY: release
release: clean build ## Prepare for release (clean + build)
	@echo "Ready for release. Don't forget to:"
	@echo "  1. Update version in pyproject.toml"
	@echo "  2. Create git tag"
	@echo "  3. Push to PyPI"