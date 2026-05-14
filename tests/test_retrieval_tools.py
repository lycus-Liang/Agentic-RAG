import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("LLM_API_KEY", "test-key")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.agent import retrieval_tools


class FakeDoc:
    def __init__(self, image_path, doc_id="doc-1", score=0.8):
        self.id = doc_id
        self.score = score
        self.payload = {
            "source_doc": "sample.pdf",
            "page_num": 3,
            "markdown_text": "文本证据" * 500,
            "proxy_descriptions": ["视觉描述" * 200],
            "extracted_crop_paths": [image_path],
        }


class FakeSearcher:
    def __init__(self, image_path):
        self.image_path = image_path
        self.calls = []

    def search(self, query, top_k=None, text_only=False, retrieval_mode="hybrid", search_style="auto"):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "text_only": text_only,
            "retrieval_mode": retrieval_mode,
            "search_style": search_style,
        })
        self.last_search_metadata = {
            "effective_search_style": search_style,
            "effective_strategies": {
                "use_dense": search_style != "exact",
                "use_sparse": True,
                "use_hyde": search_style == "expanded",
                "use_vision": retrieval_mode in {"vision", "hybrid"},
                "use_reranker": True,
            },
            "routes": ["bge_sparse"],
            "fusion": "RRF",
            "notes": [],
        }
        return [FakeDoc(self.image_path)]


def make_informative_image(path):
    image = Image.new("RGB", (160, 160), "white")
    for x in range(30, 130):
        for y in range(30, 130):
            image.putpixel((x, y), (34, 139, 34))
    image.save(path)
    return path


def test_tool_schemas_include_modal_tools():
    schemas = retrieval_tools.build_retrieval_tool_schemas(text_only=False)
    names = [schema["function"]["name"] for schema in schemas]

    assert "retrieve_text" in names
    assert "retrieve_hybrid" in names
    text_tool = next(schema for schema in schemas if schema["function"]["name"] == "retrieve_text")
    search_style = text_tool["function"]["parameters"]["properties"]["search_style"]
    assert search_style["enum"] == ["auto", "exact", "semantic", "expanded"]
    if retrieval_tools.config_dict["online"]["retrieval_strategies"].get("use_vision"):
        assert "retrieve_vision" in names


def test_tool_schemas_text_only_exposes_only_text_tool():
    schemas = retrieval_tools.build_retrieval_tool_schemas(text_only=True)

    assert [schema["function"]["name"] for schema in schemas] == ["retrieve_text"]


def test_execute_retrieval_tool_routes_modes_and_truncates_output():
    old_searcher = retrieval_tools._searcher
    with tempfile.TemporaryDirectory() as tmp:
        fake_searcher = FakeSearcher(make_informative_image(os.path.join(tmp, "crop.png")))
        retrieval_tools._searcher = fake_searcher
        try:
            text_result = retrieval_tools.execute_retrieval_tool(
                "retrieve_text",
                {"query": "文本", "search_style": "exact"},
            )
            vision_result = retrieval_tools.execute_retrieval_tool("retrieve_vision", {"query": "图表"})
            hybrid_result = retrieval_tools.execute_retrieval_tool("retrieve_hybrid", {"query": "综合"})
        finally:
            retrieval_tools._searcher = old_searcher

    assert [call["retrieval_mode"] for call in fake_searcher.calls] == ["text", "vision", "hybrid"]
    assert [call["search_style"] for call in fake_searcher.calls] == ["exact", "auto", "auto"]
    assert text_result["mode"] == "text"
    assert text_result["search_style"] == "exact"
    assert text_result["effective_strategies"]["use_sparse"] is True
    assert text_result["fusion"] == "RRF"
    assert vision_result["mode"] == "vision"
    assert hybrid_result["mode"] == "hybrid"
    assert len(text_result["documents"][0]["markdown_text"]) == retrieval_tools.MAX_TOOL_TEXT_CHARS
    assert len(text_result["documents"][0]["proxy_descriptions"][0]) == retrieval_tools.MAX_TOOL_PROXY_CHARS


def test_execute_retrieval_tool_forces_text_mode_when_text_only():
    old_searcher = retrieval_tools._searcher
    with tempfile.TemporaryDirectory() as tmp:
        fake_searcher = FakeSearcher(make_informative_image(os.path.join(tmp, "crop.png")))
        retrieval_tools._searcher = fake_searcher
        try:
            result = retrieval_tools.execute_retrieval_tool(
                "retrieve_hybrid",
                {"query": "综合"},
                text_only=True,
            )
        finally:
            retrieval_tools._searcher = old_searcher

    assert result["mode"] == "text"
    assert fake_searcher.calls[0]["text_only"] is True
    assert fake_searcher.calls[0]["retrieval_mode"] == "text"


def test_execute_retrieval_tool_normalizes_unknown_search_style():
    old_searcher = retrieval_tools._searcher
    with tempfile.TemporaryDirectory() as tmp:
        fake_searcher = FakeSearcher(make_informative_image(os.path.join(tmp, "crop.png")))
        retrieval_tools._searcher = fake_searcher
        try:
            result = retrieval_tools.execute_retrieval_tool(
                "retrieve_text",
                {"query": "文本", "search_style": "unknown"},
            )
        finally:
            retrieval_tools._searcher = old_searcher

    assert result["search_style"] == "auto"
    assert fake_searcher.calls[0]["search_style"] == "auto"


if __name__ == "__main__":
    test_tool_schemas_include_modal_tools()
    test_tool_schemas_text_only_exposes_only_text_tool()
    test_execute_retrieval_tool_routes_modes_and_truncates_output()
    test_execute_retrieval_tool_forces_text_mode_when_text_only()
    test_execute_retrieval_tool_normalizes_unknown_search_style()
    print("test_retrieval_tools passed")
