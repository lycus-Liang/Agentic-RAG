import yaml
from typing import Any, Dict, List

from src.utils.image_quality import filter_informative_images


with open("./configs/config.yaml", "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)


_searcher = None

TOOL_TO_MODE = {
    "retrieve_text": "text",
    "retrieve_vision": "vision",
    "retrieve_hybrid": "hybrid",
}

TOOL_DESCRIPTIONS = {
    "retrieve_text": "Search text embeddings and sparse textual signals for passages that answer the query.",
    "retrieve_vision": "Search visual page/image embeddings when layout, diagrams, crops, or scanned-page visual evidence matter.",
    "retrieve_hybrid": "Search both text and visual routes when the best modality is uncertain or evidence may be multimodal.",
}

SEARCH_STYLES = {"auto", "exact", "semantic", "expanded"}

MAX_TOOL_TEXT_CHARS = 800
MAX_TOOL_PROXY_CHARS = 400


def get_searcher():
    """延迟加载检索模型，避免服务启动阶段因模型下载失败而崩溃。"""
    global _searcher
    if _searcher is None:
        from src.retrieval.searcher import Searcher
        _searcher = Searcher(config_dict=config_dict)
    return _searcher


def build_retrieval_tool_schemas(text_only: bool = False) -> List[Dict[str, Any]]:
    """Return OpenAI-compatible retrieval tool schemas available for this request."""
    strategies = config_dict.get("online", {}).get("retrieval_strategies", {})
    tool_names = ["retrieve_text"]

    if not text_only:
        if strategies.get("use_vision", False):
            tool_names.append("retrieve_vision")
        tool_names.append("retrieve_hybrid")

    schemas = []
    for name in tool_names:
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The concise search query to send to the retrieval index.",
                        },
                        "search_style": {
                            "type": "string",
                            "enum": ["auto", "exact", "semantic", "expanded"],
                            "description": (
                                "Retrieval style chosen by the agent. auto follows config; "
                                "exact favors precise lexical matching; semantic favors conceptual matching; "
                                "expanded allows query expansion only when enabled by config."
                            ),
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        })
    return schemas


def _format_document_for_tool(doc) -> Dict[str, Any]:
    payload = getattr(doc, "payload", {}) or {}
    markdown_text = payload.get("markdown_text", "") or ""
    proxy_descriptions = []
    for desc in payload.get("proxy_descriptions", []) or []:
        proxy_descriptions.append(str(desc)[:MAX_TOOL_PROXY_CHARS])
    image_paths, proxy_descriptions = filter_informative_images(
        list(payload.get("extracted_crop_paths", []) or []),
        proxy_descriptions,
    )

    return {
        "id": str(getattr(doc, "id", "")),
        "score": float(getattr(doc, "score", 0.0) or 0.0),
        "source_doc": payload.get("source_doc", "unknown"),
        "page_num": payload.get("page_num", 0),
        "markdown_text": str(markdown_text)[:MAX_TOOL_TEXT_CHARS],
        "proxy_descriptions": proxy_descriptions[:3],
        "image_paths": image_paths[:3],
    }


def run_retrieval_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    text_only: bool = False,
) -> tuple[Dict[str, Any], List[Any]]:
    """Execute one retrieval tool call and return compact payload plus raw documents."""
    if tool_name not in TOOL_TO_MODE:
        raise ValueError(f"unknown retrieval tool: {tool_name}")

    query = str((arguments or {}).get("query", "")).strip()
    if not query:
        raise ValueError(f"{tool_name} requires a non-empty query")

    search_style = str((arguments or {}).get("search_style", "auto")).strip().lower() or "auto"
    if search_style not in SEARCH_STYLES:
        search_style = "auto"

    mode = "text" if text_only else TOOL_TO_MODE[tool_name]
    searcher = get_searcher()
    documents = searcher.search(
        query,
        text_only=text_only,
        retrieval_mode=mode,
        search_style=search_style,
    )
    formatted_docs = [_format_document_for_tool(doc) for doc in documents]
    metadata = getattr(searcher, "last_search_metadata", {}) or {}

    return {
        "tool_name": tool_name,
        "query": query,
        "mode": mode,
        "search_style": search_style,
        "effective_search_style": metadata.get("effective_search_style", search_style),
        "effective_strategies": metadata.get("effective_strategies", {}),
        "routes": metadata.get("routes", []),
        "fusion": metadata.get("fusion", ""),
        "strategy_notes": metadata.get("notes", []),
        "hit_count": len(documents),
        "documents": formatted_docs,
    }, documents


def execute_retrieval_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    text_only: bool = False,
) -> Dict[str, Any]:
    """Execute one retrieval tool call and return a compact JSON-serializable payload."""
    result, _ = run_retrieval_tool(tool_name, arguments, text_only=text_only)
    return result
