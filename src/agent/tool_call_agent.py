import json
import yaml
from typing import Any, Dict, List

from src.agent.retrieval_tools import build_retrieval_tool_schemas, run_retrieval_tool
from src.agent.state import GraphState
from src.utils.llm_client import LLMClient
from src.utils.image_quality import filter_informative_images


with open("./configs/config.yaml", "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", "r", encoding="utf-8") as f:
    prompts_dict = yaml.safe_load(f)


workflow_config = config_dict.get("online", {}).get("workflow", {})
tool_prompts = prompts_dict.get("agent_tool_caller", {})
MAX_TOOL_CALLS = workflow_config.get("max_plan_steps", 4)
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
    return _llm_client


def _dedupe_documents(existing, incoming):
    seen = set()
    merged = []
    for doc in list(existing or []) + list(incoming or []):
        doc_id = getattr(doc, "id", None)
        payload = getattr(doc, "payload", {}) or {}
        key = str(doc_id) if doc_id is not None else str(payload.get("markdown_text", ""))[:120]
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def _collect_image_paths(documents) -> List[str]:
    image_paths = []
    for doc in documents or []:
        payload = getattr(doc, "payload", {}) or {}
        valid_paths, _ = filter_informative_images(
            payload.get("extracted_crop_paths", []),
            payload.get("proxy_descriptions", []),
        )
        for path in valid_paths:
            if path and path not in image_paths:
                image_paths.append(path)
    return image_paths


def _build_evidence_summary(result: Dict[str, Any]) -> str:
    chunks = []
    for doc in result.get("documents", [])[:3]:
        source = doc.get("source_doc", "unknown")
        page = doc.get("page_num", 0)
        text = doc.get("markdown_text", "")
        if text:
            chunks.append(f"[{source} | 第 {page} 页]\n{text}")
        proxy_descriptions = doc.get("proxy_descriptions", []) or []
        if proxy_descriptions:
            chunks.append("视觉代理描述：" + "\n".join(proxy_descriptions[:2]))
        image_paths = doc.get("image_paths", []) or []
        if image_paths:
            chunks.append(f"关联图片：{', '.join(image_paths[:3])}")
    return "\n\n".join(chunks) or "无可用证据摘要"


def _parse_tool_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        name = function.get("name", "unknown")
        raise ValueError(f"tool call arguments for {name} are not valid JSON: {raw_arguments}") from e
    if not isinstance(arguments, dict):
        name = function.get("name", "unknown")
        raise ValueError(f"tool call arguments for {name} must be a JSON object")
    return arguments


def _default_plan(question: str, text_only: bool = False) -> List[Dict[str, Any]]:
    return [{
        "id": "step_1",
        "sub_question": question,
        "mode": "text" if text_only else "hybrid",
        "depends_on": [],
        "purpose": "直接检索回答用户问题所需的核心证据",
    }]


def _format_prior_evidence(step_results: List[Dict[str, Any]], depends_on: List[str]) -> str:
    if not step_results:
        return "无前序证据"

    depends_on_set = set(depends_on or [])
    chunks = []
    for idx, result in enumerate(step_results, start=1):
        step = result.get("step", {})
        step_id = step.get("id") or f"step_{idx}"
        if depends_on_set and step_id not in depends_on_set:
            continue

        summary = result.get("evidence_summary") or result.get("evidence", {}).get("evidence_summary", "")
        query = result.get("resolved_query") or result.get("query") or step.get("sub_question", "")
        chunks.append(
            f"[{step_id} | mode={result.get('mode', step.get('mode', 'hybrid'))} | query={query}]\n"
            f"{str(summary)[:1200]}"
        )

    return "\n\n".join(chunks).strip() or "无可用前序证据"


def _compatible_tool_names(planned_mode: str, available_tool_names: set) -> set:
    planned_mode = (planned_mode or "hybrid").lower()
    preferred = {
        "text": {"retrieve_text", "retrieve_hybrid"},
        "vision": {"retrieve_vision", "retrieve_hybrid"},
        "hybrid": {"retrieve_text", "retrieve_vision", "retrieve_hybrid"},
    }.get(planned_mode, {"retrieve_hybrid"})
    compatible = preferred & available_tool_names
    return compatible or available_tool_names


def _step_result_from_tool(
    result: Dict[str, Any],
    raw_documents,
    tool_call: Dict[str, Any],
    tool_index: int,
    planned_step: Dict[str, Any],
    prior_evidence: str,
) -> Dict[str, Any]:
    mode = result.get("mode", "hybrid")
    query = result.get("query", "")
    search_style = result.get("search_style", "auto")
    evidence_summary = _build_evidence_summary(result)
    step = dict(planned_step or {})
    step.setdefault("id", f"step_{tool_index}")
    step.setdefault("sub_question", query)
    step.setdefault("mode", mode)
    step.setdefault("depends_on", [])
    step.setdefault("purpose", f"模型通过 {result.get('tool_name')} 工具选择的检索动作")
    return {
        "step": step,
        "query": query,
        "resolved_query": query,
        "dependency_context": prior_evidence,
        "attempts": [{
            "tool_call_id": tool_call.get("id"),
            "tool_name": result.get("tool_name"),
            "query": query,
            "mode": mode,
            "search_style": search_style,
            "effective_search_style": result.get("effective_search_style", search_style),
            "effective_strategies": result.get("effective_strategies", {}),
            "routes": result.get("routes", []),
            "fusion": result.get("fusion", ""),
            "hit_count": result.get("hit_count", 0),
        }],
        "mode": mode,
        "relevant": bool(raw_documents),
        "documents": raw_documents,
        "evidence": {
            "mode": mode,
            "search_style": search_style,
            "effective_search_style": result.get("effective_search_style", search_style),
            "effective_strategies": result.get("effective_strategies", {}),
            "routes": result.get("routes", []),
            "fusion": result.get("fusion", ""),
            "hit_count": result.get("hit_count", 0),
            "image_paths": _collect_image_paths(raw_documents),
            "image_count": len(_collect_image_paths(raw_documents)),
            "sources": [
                {
                    "source_doc": doc.get("source_doc", "unknown"),
                    "page_num": doc.get("page_num", 0),
                }
                for doc in result.get("documents", [])
            ],
            "evidence_summary": evidence_summary,
        },
        "image_paths": _collect_image_paths(raw_documents),
        "evidence_summary": evidence_summary,
        "resolved_facts": {},
    }


def tool_call_agent_node(state: GraphState) -> Dict:
    """Node: execute the retrieval plan through native tool calls."""
    question = state["question"]
    original_q = state.get("original_question", question)
    text_only = state.get("text_only", False)
    debug = state.get("debug", False)
    max_tool_calls = max(1, int(MAX_TOOL_CALLS or 4))
    plan = state.get("retrieval_plan") or _default_plan(question, text_only=text_only)
    plan = plan[:max_tool_calls]

    print("--- 🧰 Agent 正在按检索计划通过 tool call 执行检索 ---")

    tools = build_retrieval_tool_schemas(text_only=text_only)
    allowed_tool_names = {tool["function"]["name"] for tool in tools}
    sys_prompt = tool_prompts.get("system_prompt", "你是一个必须通过工具检索证据的 RAG Agent。")

    documents = []
    step_results = []
    tool_trace = list(state.get("tool_trace", []))
    agent_trace = list(state.get("agent_trace", []))
    tool_call_count = 0
    stop_reason = "completed_plan"

    for step_index, step in enumerate(plan, start=1):
        step = dict(step)
        planned_mode = step.get("mode", "hybrid")
        planned_query = step.get("sub_question", question)
        prior_evidence = _format_prior_evidence(step_results, step.get("depends_on", []))
        compatible_tool_names = _compatible_tool_names(planned_mode, allowed_tool_names)
        user_prompt = tool_prompts.get("user_template", "{question}").format(
            question=question,
            original_q=original_q,
            text_only=text_only,
            max_tool_calls=max_tool_calls,
            available_tools=", ".join(sorted(allowed_tool_names)),
            compatible_tools=", ".join(sorted(compatible_tool_names)),
            plan_json=json.dumps({"steps": plan}, ensure_ascii=False),
            current_step_json=json.dumps(step, ensure_ascii=False),
            step_index=step_index,
            total_steps=len(plan),
            prior_evidence=prior_evidence,
        )
        messages = [{"role": "user", "content": user_prompt}]
        assistant_message = _get_llm_client().chat_with_tools(
            system_prompt=sys_prompt,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            content = str(assistant_message.get("content", "")).strip()
            raise RuntimeError(
                f"模型未为计划步骤 {step.get('id', f'step_{step_index}')} "
                "通过原生 tool_calls 调用检索工具。"
                f"模型返回内容: {content[:160]}"
            )

        messages.append({
            "role": "assistant",
            "content": assistant_message.get("content", ""),
            "tool_calls": tool_calls,
        })

        if len(tool_calls) > 1:
            agent_trace.append(
                f"计划步骤 {step.get('id', f'step_{step_index}')} 返回多个 tool_call，"
                "仅执行第一个。"
            )

        tool_call = tool_calls[0]
        tool_call_id = tool_call.get("id")
        function = tool_call.get("function") or {}
        tool_name = function.get("name")
        if not tool_call_id:
            raise RuntimeError("模型返回的 tool_call 缺少 id，无法回传工具结果。")
        if tool_name not in allowed_tool_names:
            raise RuntimeError(f"模型请求了当前不可用的检索工具：{tool_name}")
        if tool_name not in compatible_tool_names:
            raise RuntimeError(
                f"模型为计划步骤 {step.get('id', f'step_{step_index}')} 请求了不兼容工具："
                f"{tool_name}。计划模态={planned_mode}，允许工具={sorted(compatible_tool_names)}"
            )

        arguments = _parse_tool_arguments(tool_call)
        result, raw_documents = run_retrieval_tool(tool_name, arguments, text_only=text_only)
        tool_call_count += 1
        documents = _dedupe_documents(documents, raw_documents)
        step_results.append(
            _step_result_from_tool(
                result,
                raw_documents,
                tool_call,
                tool_call_count,
                step,
                prior_evidence,
            )
        )

        trace_item = {
            "round": len(step_results),
            "step_id": step.get("id", f"step_{step_index}"),
            "step_index": step_index,
            "planned_mode": planned_mode,
            "planned_query": planned_query,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "query": result.get("query", ""),
            "mode": result.get("mode", "hybrid"),
            "actual_query": result.get("query", ""),
            "actual_mode": result.get("mode", "hybrid"),
            "search_style": result.get("search_style", "auto"),
            "effective_search_style": result.get("effective_search_style", result.get("search_style", "auto")),
            "effective_strategies": result.get("effective_strategies", {}),
            "routes": result.get("routes", []),
            "fusion": result.get("fusion", ""),
            "strategy_notes": result.get("strategy_notes", []),
            "hit_count": result.get("hit_count", 0),
        }
        tool_trace.append(trace_item)
        if debug:
            print(
                f"  🧰 step {step_index}/{len(plan)} {step.get('id', '')}: "
                f"{tool_name} | planned={planned_mode}:{planned_query} | "
                f"actual={result.get('mode', 'hybrid')}:{result.get('query', '')} | "
                f"style={result.get('search_style', 'auto')}->{result.get('effective_search_style', result.get('search_style', 'auto'))} | "
                f"hits={result.get('hit_count', 0)}"
            )
        agent_trace.append(
            f"步骤 {step_index}/{len(plan)} tool_call: {tool_name}, "
            f"planned_mode={planned_mode}, query={result.get('query', '')}, "
            f"style={result.get('search_style', 'auto')}, "
            f"hits={result.get('hit_count', 0)}"
        )

    return {
        "original_question": original_q,
        "retrieval_plan": plan,
        "documents": documents,
        "step_results": step_results,
        "tool_trace": tool_trace,
        "tool_call_count": tool_call_count,
        "tool_agent_stop_reason": stop_reason,
        "agent_trace": agent_trace,
        "retrieval_error": "",
    }
