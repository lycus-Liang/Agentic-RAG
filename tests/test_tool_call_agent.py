import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_API_KEY", "test-key")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agent import tool_call_agent


class FakeDoc:
    id = "doc-1"
    score = 0.9
    payload = {
        "source_doc": "sample.pdf",
        "page_num": 2,
        "markdown_text": "有效文本证据",
        "proxy_descriptions": ["版面视觉描述"],
        "extracted_crop_paths": ["/tmp/crop.jpg"],
    }


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_with_tools(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            return {"role": "assistant", "content": "READY", "tool_calls": []}
        return self.responses.pop(0)


def tool_call(name="retrieve_text", arguments='{"query":"原文锚点"}', call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def fake_run_retrieval_tool(tool_name, arguments, text_only=False):
    mode = "text" if text_only else {
        "retrieve_text": "text",
        "retrieve_vision": "vision",
        "retrieve_hybrid": "hybrid",
    }[tool_name]
    search_style = arguments.get("search_style", "auto")
    return {
        "tool_name": tool_name,
        "query": arguments["query"],
        "mode": mode,
        "search_style": search_style,
        "effective_search_style": search_style,
        "effective_strategies": {
            "use_dense": search_style != "exact",
            "use_sparse": True,
            "use_hyde": search_style == "expanded",
            "use_vision": mode in {"vision", "hybrid"},
            "use_reranker": True,
        },
        "routes": ["bge_sparse"],
        "fusion": "RRF",
        "strategy_notes": [],
        "hit_count": 1,
        "documents": [{
            "source_doc": "sample.pdf",
            "page_num": 2,
            "score": 0.9,
            "markdown_text": "有效文本证据",
            "proxy_descriptions": ["版面视觉描述"],
            "image_paths": ["/tmp/crop.jpg"],
        }],
    }, [FakeDoc()]


def with_fakes(llm, max_tool_calls=4):
    old_llm = tool_call_agent._llm_client
    old_runner = tool_call_agent.run_retrieval_tool
    old_max = tool_call_agent.MAX_TOOL_CALLS
    tool_call_agent._llm_client = llm
    tool_call_agent.run_retrieval_tool = fake_run_retrieval_tool
    tool_call_agent.MAX_TOOL_CALLS = max_tool_calls
    return old_llm, old_runner, old_max


def restore_fakes(saved):
    old_llm, old_runner, old_max = saved
    tool_call_agent._llm_client = old_llm
    tool_call_agent.run_retrieval_tool = old_runner
    tool_call_agent.MAX_TOOL_CALLS = old_max


def two_step_plan():
    return [
        {
            "id": "step_1",
            "sub_question": "找到原文锚点",
            "mode": "text",
            "depends_on": [],
            "purpose": "定位文本证据",
        },
        {
            "id": "step_2",
            "sub_question": "基于 step_1 的证据查找相关图片",
            "mode": "hybrid",
            "depends_on": ["step_1"],
            "purpose": "补充图文证据",
        },
    ]


def test_tool_call_agent_executes_single_planned_step():
    llm = FakeLLM([
        {"role": "assistant", "content": "", "tool_calls": [
            tool_call(arguments='{"query":"原文锚点","search_style":"exact"}')
        ]},
    ])
    saved = with_fakes(llm)
    try:
        result = tool_call_agent.tool_call_agent_node({
            "question": "题目",
            "text_only": False,
            "retrieval_plan": [two_step_plan()[0]],
        })
    finally:
        restore_fakes(saved)

    assert result["tool_call_count"] == 1
    assert result["tool_agent_stop_reason"] == "completed_plan"
    assert result["documents"]
    assert result["step_results"]
    assert result["tool_trace"][0]["tool_name"] == "retrieve_text"
    assert result["tool_trace"][0]["step_id"] == "step_1"
    assert result["tool_trace"][0]["planned_mode"] == "text"
    assert result["tool_trace"][0]["planned_query"] == "找到原文锚点"
    assert result["tool_trace"][0]["search_style"] == "exact"
    assert result["tool_trace"][0]["effective_search_style"] == "exact"
    assert result["tool_trace"][0]["effective_strategies"]["use_sparse"] is True
    assert result["tool_trace"][0]["fusion"] == "RRF"
    assert len(llm.calls) == 1


def test_tool_call_agent_executes_multi_step_plan():
    llm = FakeLLM([
        {"role": "assistant", "content": "", "tool_calls": [tool_call(call_id="call-1")]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(
                name="retrieve_hybrid",
                arguments='{"query":"竹子 图片","search_style":"semantic"}',
                call_id="call-2",
            )],
        },
    ])
    saved = with_fakes(llm)
    try:
        result = tool_call_agent.tool_call_agent_node({
            "question": "题目",
            "text_only": False,
            "retrieval_plan": two_step_plan(),
        })
    finally:
        restore_fakes(saved)

    assert result["tool_call_count"] == 2
    assert len(result["step_results"]) == 2
    assert [trace["step_id"] for trace in result["tool_trace"]] == ["step_1", "step_2"]
    assert result["tool_trace"][1]["tool_name"] == "retrieve_hybrid"
    assert result["tool_trace"][1]["search_style"] == "semantic"
    assert "有效文本证据" in result["step_results"][1]["dependency_context"]
    assert len(llm.calls) == 2


def test_tool_call_agent_errors_if_first_response_has_no_tool_calls():
    llm = FakeLLM([
        {"role": "assistant", "content": "READY", "tool_calls": []},
    ])
    saved = with_fakes(llm)
    try:
        try:
            tool_call_agent.tool_call_agent_node({
                "question": "题目",
                "text_only": False,
                "retrieval_plan": [two_step_plan()[0]],
            })
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "tool_calls" in str(e)
    finally:
        restore_fakes(saved)


def test_tool_call_agent_rejects_malformed_tool_arguments():
    llm = FakeLLM([
        {"role": "assistant", "content": "", "tool_calls": [tool_call(arguments="{bad json")]},
    ])
    saved = with_fakes(llm)
    try:
        try:
            tool_call_agent.tool_call_agent_node({
                "question": "题目",
                "text_only": False,
                "retrieval_plan": [two_step_plan()[0]],
            })
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "valid JSON" in str(e)
    finally:
        restore_fakes(saved)


def test_tool_call_agent_text_only_allows_only_text_tool():
    llm = FakeLLM([
        {"role": "assistant", "content": "", "tool_calls": [tool_call()]},
    ])
    saved = with_fakes(llm)
    try:
        result = tool_call_agent.tool_call_agent_node({
            "question": "题目",
            "text_only": True,
            "retrieval_plan": [{
                "id": "step_1",
                "sub_question": "题目",
                "mode": "text",
                "depends_on": [],
                "purpose": "纯文本检索",
            }],
        })
    finally:
        restore_fakes(saved)

    assert result["tool_call_count"] == 1
    assert result["tool_trace"][0]["actual_mode"] == "text"
    tool_names = [tool["function"]["name"] for tool in llm.calls[0]["tools"]]
    assert tool_names == ["retrieve_text"]
    assert len(llm.calls) == 1


if __name__ == "__main__":
    test_tool_call_agent_executes_single_planned_step()
    test_tool_call_agent_executes_multi_step_plan()
    test_tool_call_agent_errors_if_first_response_has_no_tool_calls()
    test_tool_call_agent_rejects_malformed_tool_arguments()
    test_tool_call_agent_text_only_allows_only_text_tool()
    print("test_tool_call_agent passed")
