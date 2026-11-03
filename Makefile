# SentinelMCP developer commands. Requires: uv, docker (for dev/down/test-int).
.DEFAULT_GOAL := help
.PHONY: help install dev down test test-int attacks lint fmt policy-check bench migrate

UV     ?= uv
RUN    := $(UV) run
PYTEST := $(RUN) pytest

# pytest exit code 5 = "no tests collected"; tolerated only for suites not written yet.
define pytest_allow_empty
	$(PYTEST) $(1) || { rc=$$?; if [ $$rc -eq 5 ]; then echo "(no tests collected yet)"; else exit $$rc; fi; }
endef

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install project + dev extras into .venv
	$(UV) sync --extra dev

dev: ## Start local stack (docker compose)
	docker compose up -d --wait

down: ## Stop local stack
	docker compose down

test: ## Unit tests (no external services)
	$(PYTEST) tests/unit

test-int: ## Integration tests (needs `make dev`)
	$(call pytest_allow_empty,-m integration tests/integration)

attacks: ## Attack regression suite
	$(call pytest_allow_empty,-m attack tests/attack_regression)

lint: ## ruff + format check + mypy
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy gateway tools_servers tests

fmt: ## Auto-format and fix lint
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

policy-check: ## Validate the policy file and replay all policy scenarios
	$(RUN) python -m gateway.policy validate
	@for f in policies/scenarios/*.jsonl; do echo "== $$f"; $(RUN) python -m gateway.policy dry-run --scenario $$f || exit 1; done

bench: ## Benchmark a decider on a split (Phase 3)
	@echo "bench: not implemented until Phase 3 (eval/benchmark.py)"; exit 1

migrate: ## Apply DB migrations to SENTINEL_DATABASE_URL
	$(RUN) alembic upgrade head
