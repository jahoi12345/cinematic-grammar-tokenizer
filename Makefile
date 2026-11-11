.PHONY: setup setup-ml shotbench test
setup:; uv sync --group dev
setup-ml:; uv sync --group dev --group ml --group demo --group vlm
shotbench:; uv run python scripts/download_shotbench.py
test:; uv run pytest -q
