import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

qdrant_client = types.ModuleType("qdrant_client")
qdrant_client.QdrantClient = object
qdrant_http = types.ModuleType("qdrant_client.http")
qdrant_http.models = types.SimpleNamespace()
flag_embedding = types.ModuleType("FlagEmbedding")
flag_embedding.FlagReranker = object
query_encoder = types.ModuleType("src.retrieval.query_encoder")
query_encoder.QueryEncoder = object

sys.modules.setdefault("qdrant_client", qdrant_client)
sys.modules.setdefault("qdrant_client.http", qdrant_http)
sys.modules.setdefault("FlagEmbedding", flag_embedding)
sys.modules.setdefault("src.retrieval.query_encoder", query_encoder)

from src.retrieval.searcher import Searcher


def make_searcher(strategies):
    searcher = Searcher.__new__(Searcher)
    searcher.strategies = strategies
    searcher.use_reranker = strategies.get("use_reranker", False)
    return searcher


def test_exact_prefers_sparse_and_disables_hyde():
    searcher = make_searcher({
        "use_dense": True,
        "use_sparse": True,
        "use_hyde": True,
        "use_vision": True,
        "use_reranker": True,
    })

    effective = searcher._resolve_effective_strategies("exact")

    assert effective["use_dense"] is False
    assert effective["use_sparse"] is True
    assert effective["use_hyde"] is False
    assert effective["effective_search_style"] == "exact"


def test_expanded_is_capped_by_config_when_hyde_disabled():
    searcher = make_searcher({
        "use_dense": True,
        "use_sparse": True,
        "use_hyde": False,
        "use_vision": True,
        "use_reranker": False,
    })

    effective = searcher._resolve_effective_strategies("expanded")

    assert effective["use_hyde"] is False
    assert effective["requested_search_style"] == "expanded"
    assert effective["effective_search_style"] == "semantic"
    assert "expanded_hyde_disabled_by_config" in effective["notes"]


def test_unknown_style_falls_back_to_auto():
    searcher = make_searcher({
        "use_dense": True,
        "use_sparse": False,
        "use_hyde": False,
    })

    effective = searcher._resolve_effective_strategies("invalid")

    assert effective["requested_search_style"] == "auto"
    assert effective["use_dense"] is True
    assert effective["use_sparse"] is False


if __name__ == "__main__":
    test_exact_prefers_sparse_and_disables_hyde()
    test_expanded_is_capped_by_config_when_hyde_disabled()
    test_unknown_style_falls_back_to_auto()
    print("test_search_styles passed")
