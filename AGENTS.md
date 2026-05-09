# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`, split by responsibility: `src/agent/` for workflow and routing, `src/ingestion/` for PDF/JSON parsing, `src/retrieval/` for search, `src/storage/` for Qdrant integration, and `src/utils/` for shared helpers. Entry points are `main.py` for CLI build/chat flows and `api_server.py` for the FastAPI service. Config files live in `configs/`, helper shell scripts in `scripts/`, exploratory Streamlit tools in `notebooks/`, and tests in `tests/` plus the root-level `test_api.py`.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`.
Use `python main.py build --file ./data/raw/sample.pdf --type pdf` to ingest a single PDF, or `python main.py chat --text-only` for text-only retrieval/chat.
Run the API locally with `python api_server.py`.
Launch the Streamlit tools with `streamlit run notebooks/web_ui.py --server.port 6006` or `streamlit run notebooks/align_viewer.py --server.port 8501`.
Start required services from `scripts/`: `./start_db.sh`, `./start_omniparse.sh`, and optionally `./start_local_llm.sh`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions/files, PascalCase for classes, and small focused modules under `src/`. Keep new CLI flags and config keys descriptive and aligned with current naming such as `text_only`, `collection_name`, and `batch_size`. No formatter or linter is configured in this repo, so keep imports tidy and match surrounding code style before submitting changes.

## Testing Guidelines
Use `python tests/test_agent.py` for workflow checks and `python tests/test_qdrant_points.py` to validate Qdrant payload/vector integrity after ingestion. `python test_api.py` is the quick API smoke test. Name new tests `test_*.py` and keep them runnable as standalone scripts unless you also introduce a formal test runner. Prefer targeted checks around ingestion, retrieval, and API behavior.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages, often in Chinese, for example `更新测试pipeline` and `添加api调用rag的方式`. Keep commit subjects brief and specific to one change. PRs should describe the user-visible effect, list required config or service prerequisites, link related tasks, and include screenshots or sample requests/responses when changing the Streamlit UI or API outputs.

## Security & Configuration Tips
Do not commit secrets from `.env`. Document any required changes to `configs/config.yaml` and mention external dependencies such as HuggingFace mirrors, Qdrant, Omniparse, or local vLLM when they affect setup or testing.
