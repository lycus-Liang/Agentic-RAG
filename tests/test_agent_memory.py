import os
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent import memory
from src.agent.memory import AgentMemoryStore


def make_store():
    tmp = tempfile.TemporaryDirectory()
    store = AgentMemoryStore(os.path.join(tmp.name, "memory.sqlite3"))
    return tmp, store


def sample_record(question="如何查找包含竹子的图片？", reward=0.8):
    return {
        "session_id": "s1",
        "question": question,
        "profile": "用户需要查找具有明确视觉对象的图片。",
        "insight": "应优先选择 retrieve_vision 或 retrieve_hybrid，并使用 semantic search_style。",
        "utility": "遇到相似视觉对象查找问题时优先使用视觉或混合检索。",
        "tags": ["retrieve_vision", "semantic", "image"],
        "quality": "success",
        "plan_json": {"steps": [{"id": "step_1", "mode": "vision"}]},
        "tool_trace_json": [{"tool_name": "retrieve_vision", "search_style": "semantic"}],
        "answer_summary": "找到了相关图片。",
        "reward": reward,
        "reward_source": "heuristic",
    }


def test_memory_store_writes_and_retrieves_experience():
    tmp, store = make_store()
    try:
        result = store.add_or_update_experience(sample_record())
        memories = store.retrieve("我想找一张有竹子的图片", session_id="s1", top_k=3, min_reward=0.3)
    finally:
        tmp.cleanup()

    assert result["action"] == "add"
    assert result["id"]
    assert memories
    assert "视觉" in memories[0]["profile"] or "图片" in memories[0]["question"]


def test_memory_store_filters_low_reward():
    tmp, store = make_store()
    try:
        store.add_or_update_experience(sample_record(reward=-0.5))
        memories = store.retrieve("竹子 图片", session_id="s1", top_k=3, min_reward=0.3)
    finally:
        tmp.cleanup()

    assert memories == []


def test_memory_feedback_updates_reward():
    tmp, store = make_store()
    try:
        result = store.add_or_update_experience(sample_record(reward=0.4))
        ok = store.apply_feedback(result["id"], session_id="s1", score=-1.0, comment="结果不相关")
        memories = store.retrieve("竹子 图片", session_id="s1", top_k=3, min_reward=0.3)
    finally:
        tmp.cleanup()

    assert ok is True
    assert memories == []


def test_update_memory_node_uses_fallback_reflection_without_llm():
    tmp, store = make_store()
    old_store = memory._memory_store
    old_config = dict(memory.memory_config)
    try:
        memory._memory_store = store
        memory.memory_config.update({
            "enabled": True,
            "write_after_chat": True,
            "use_llm_reflection": False,
        })
        result = memory.update_memory_node({
            "session_id": "s1",
            "question": "如何查找包含竹子的图片？",
            "generation": "答案",
            "retrieval_plan": [{"id": "step_1", "mode": "vision"}],
            "tool_trace": [{"tool_name": "retrieve_vision", "search_style": "semantic"}],
            "documents": [object()],
            "retrieval_error": "",
        })
    finally:
        memory._memory_store = old_store
        memory.memory_config.clear()
        memory.memory_config.update(old_config)
        tmp.cleanup()

    assert result["memory_record_id"]
    assert result["memory_action"] in {"add", "merge", "keep"}
    assert any("memory_" in item for item in result["memory_trace"])


if __name__ == "__main__":
    test_memory_store_writes_and_retrieves_experience()
    test_memory_store_filters_low_reward()
    test_memory_feedback_updates_reward()
    test_update_memory_node_uses_fallback_reflection_without_llm()
    print("test_agent_memory passed")
