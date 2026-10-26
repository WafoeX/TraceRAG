.PHONY: install db down run seed examples large eval lint test test-cache export-requirements
.PHONY: baseline-db baseline-stop baseline-plan baseline-db-check

install:        ## Install locked Python dependencies with uv
	uv sync --frozen --group dev

db:             ## Start Postgres + pgvector (Docker)
	docker compose up -d

down:           ## Stop the database
	docker compose down

run:            ## Run the API with autoreload
	uv run --frozen uvicorn app.main:app --reload

seed:           ## Ingest a sample doc and run a demo query
	uv run --frozen python -m scripts.seed

examples:       ## Ingest the bundled example docs (handbook, FAQ, PDF)
	uv run --frozen python -m scripts.ingest_examples

large:          ## Download a large public-domain book (Gutenberg) for scale testing
	uv run --frozen python -m scripts.fetch_large_sample

eval:           ## Measure retrieval quality (hit-rate@k + MRR) vs a vector-only baseline
	uv run --frozen python -m scripts.eval

lint:           ## Lint the codebase with ruff
	uv run --frozen ruff check .

test-cache:     ## Prepare the real tokenizer cache (network allowed only here)
	uv run --frozen python -m scripts.prepare_test_cache

test:           ## Run tests with network sockets disabled (run test-cache once first)
	uv run --frozen pytest -q

export-requirements: ## Regenerate the deployment-only requirements export
	uv export --frozen --no-dev --no-hashes --no-emit-project -o requirements.txt

baseline-db:    ## Start only the isolated baseline PostgreSQL (port 55432)
	docker compose -f compose.baseline.yml up -d --wait

baseline-stop:  ## Stop the isolated DB; preserve its volume
	docker compose -f compose.baseline.yml stop baseline-db

baseline-plan:  ## Write a no-model/no-DB capture plan; OUT must be a new directory
	test -n "$(OUT)" || (echo 'Usage: make baseline-plan OUT=runs/baseline/new-run'; exit 2)
	uv run --frozen python -m scripts.capture_baseline --output "$(OUT)"

baseline-db-check: ## Real PostgreSQL checks with fake vectors, no model calls
	test -n "$(OUT)" || (echo 'Usage: make baseline-db-check OUT=runs/infra/new-run'; exit 2)
	uv run --frozen python -m scripts.verify_baseline_database --output "$(OUT)"
