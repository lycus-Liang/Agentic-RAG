import json
import re
import yaml
from typing import Any, Dict, List

from src.agent.state import GraphState

with open("./configs/config.yaml", "r", encoding="utf-8") as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", "r", encoding="utf-8") as f:
    prompts_dict = yaml.safe_load(f)

workflow_config = config_dict.get("online", {}).get("workflow", {})
planner_prompts = prompts_dict.get("agent_planner", {})
router_prompts = prompts_dict.get("agent_router", {})
resolver_prompts = prompts_dict.get("agent_query_resolver", {})
validator_prompts = prompts_dict.get("agent_plan_validator", {})
repairer_prompts = prompts_dict.get("agent_plan_repairer", {})

MAX_STEPS = workflow_config.get("max_plan_steps", 4)
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from src.utils.llm_client import LLMClient
        _llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
    return _llm_client


def _default_plan(question: str, text_only: bool = False) -> List[Dict[str, Any]]:
    mode = "text" if text_only else "hybrid"
    return [{
        "id": "step_1",
        "sub_question": question,
        "mode": mode,
        "depends_on": [],
        "purpose": "直接检索回答用户问题所需的核心证据",
    }]


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_plan(raw_plan: Dict[str, Any], question: str, text_only: bool = False) -> List[Dict[str, Any]]:
    steps = raw_plan.get("steps", [])
    if not isinstance(steps, list):
        return _default_plan(question, text_only=text_only)

    normalized = []
    allowed_modes = {"text", "vision", "hybrid"}
    for idx, step in enumerate(steps[:MAX_STEPS], start=1):
        if not isinstance(step, dict):
            continue

        sub_question = str(step.get("sub_question", "")).strip()
        if not sub_question:
            continue

        mode = str(step.get("mode", "hybrid")).strip().lower()
        if mode not in allowed_modes:
            mode = "hybrid"
        if text_only:
            mode = "text"

        depends_on = step.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []

        normalized_step = {
            "id": str(step.get("id") or f"step_{idx}"),
            "sub_question": sub_question,
            "mode": mode,
            "depends_on": [str(item) for item in depends_on],
            "purpose": str(step.get("purpose", "检索当前子问题所需证据")).strip(),
        }

        if normalized_step["depends_on"]:
            normalized_step["sub_question"] = _guard_dependent_sub_question(
                normalized_step,
                question,
            )

        normalized.append(normalized_step)

    return normalized or _default_plan(question, text_only=text_only)


def _guard_dependent_sub_question(step: Dict[str, Any], original_question: str) -> str:
    """防止规划器在依赖步骤中提前填入前序步骤答案。"""
    sub_question = step["sub_question"]
    depends_on = step.get("depends_on", [])

    if any(dep in sub_question for dep in depends_on):
        return sub_question

    if sub_question and sub_question in original_question:
        return sub_question

    dependency_text = "、".join(depends_on)
    purpose = step.get("purpose") or "继续检索后续证据"
    return f"基于 {dependency_text} 的检索证据，{purpose}"


def _basic_plan_issues(plan: List[Dict[str, Any]], question: str) -> List[str]:
    """只做通用结构检查，不按具体题型硬拆。"""
    issues = []
    if not plan:
        return ["empty_plan"]

    step_ids = [step.get("id") for step in plan]
    if len(step_ids) != len(set(step_ids)):
        issues.append("duplicate_step_ids")

    known_ids = set(step_ids)
    for step in plan:
        step_id = step.get("id", "")
        for dep in step.get("depends_on", []):
            if dep not in known_ids:
                issues.append(f"{step_id}_depends_on_unknown_{dep}")
            if dep == step_id:
                issues.append(f"{step_id}_depends_on_self")

        if step.get("depends_on") and not any(dep in step.get("sub_question", "") for dep in step.get("depends_on", [])):
            issues.append(f"{step_id}_dependency_not_referenced")

    if len(plan) == 1 and _looks_like_multi_hop_question(question):
        issues.append("single_step_for_likely_multi_hop_question")

    return issues


def _looks_like_multi_hop_question(question: str) -> bool:
    q = question.strip()
    if len([part for part in re.split(r"[？?]", q) if part.strip()]) >= 2:
        return True

    reference_markers = ["他", "她", "其", "该", "这个", "这篇", "上述", "前者", "后者", "对应"]
    sequential_markers = ["先", "再", "然后", "之后", "基于", "结合", "根据", "找到", "找出", "确定"]
    nested_markers = ["最后", "末", "结尾", "第", "其中", "所属", "来源", "出处", "作者", "作品", "文章", "原文"]

    has_reference = any(marker in q for marker in reference_markers)
    has_sequence = sum(1 for marker in sequential_markers if marker in q) >= 1
    has_nested = sum(1 for marker in nested_markers if marker in q) >= 2
    has_many_relations = q.count("的") >= 2

    asks_property = any(token in q for token in ["什么", "谁", "哪", "多少", "如何", "为什么"])
    return has_reference or (has_sequence and has_nested) or (has_many_relations and (has_nested or asks_property))


def _validate_plan_with_llm(question: str, plan: List[Dict[str, Any]], text_only: bool = False) -> Dict[str, Any]:
    sys_prompt = validator_prompts.get("system_prompt", "你是检索计划审查器。")
    user_prompt = validator_prompts.get("user_template", "{question}").format(
        question=question,
        plan_json=json.dumps({"steps": plan}, ensure_ascii=False),
        text_only=text_only,
    )

    try:
        raw_response = _get_llm_client().generate(sys_prompt, user_prompt, temperature=0.0)
        validation = _extract_json_object(raw_response)
        return {
            "valid": bool(validation.get("valid", False)),
            "issues": validation.get("issues", []),
        }
    except Exception as e:
        return {"valid": True, "issues": [f"validator_unavailable: {e}"]}


def _repair_plan_with_llm(
    question: str,
    plan: List[Dict[str, Any]],
    issues: List[str],
    text_only: bool = False,
) -> List[Dict[str, Any]]:
    sys_prompt = repairer_prompts.get("system_prompt", "你是检索计划修复器。")
    user_prompt = repairer_prompts.get("user_template", "{question}").format(
        question=question,
        plan_json=json.dumps({"steps": plan}, ensure_ascii=False),
        issues=json.dumps(issues, ensure_ascii=False),
        text_only=text_only,
    )

    raw_response = _get_llm_client().generate(sys_prompt, user_prompt, temperature=0.0)
    return _normalize_plan(_extract_json_object(raw_response), question, text_only=text_only)


def _validate_or_repair_plan(
    question: str,
    plan: List[Dict[str, Any]],
    text_only: bool = False,
) -> Dict[str, Any]:
    basic_issues = _basic_plan_issues(plan, question)
    validation = _validate_plan_with_llm(question, plan, text_only=text_only)
    llm_issues = validation.get("issues", [])

    issues = basic_issues + [str(issue) for issue in llm_issues]
    should_repair = bool(basic_issues) or not validation.get("valid", True)
    if not should_repair:
        return {"plan": plan, "validation": {"valid": True, "issues": issues, "repaired": False}}

    try:
        repaired_plan = _repair_plan_with_llm(question, plan, issues, text_only=text_only)
        repaired_issues = _basic_plan_issues(repaired_plan, question)
        if repaired_plan and not repaired_issues:
            return {
                "plan": repaired_plan,
                "validation": {"valid": True, "issues": issues, "repaired": True},
            }
        return {
            "plan": repaired_plan or plan,
            "validation": {"valid": not repaired_issues, "issues": issues + repaired_issues, "repaired": True},
        }
    except Exception as e:
        return {
            "plan": plan,
            "validation": {"valid": False, "issues": issues + [f"repair_failed: {e}"], "repaired": False},
        }


def _format_prior_context(step_results: List[Dict[str, Any]], depends_on: List[str]) -> str:
    chunks = []
    depends_on_set = set(depends_on)

    for idx, result in enumerate(step_results, start=1):
        step = result.get("step", {})
        step_id = step.get("id") or f"step_{idx}"
        if depends_on_set and step_id not in depends_on_set:
            continue

        chunks.append(f"[{step_id} | 子问题: {result.get('query', step.get('sub_question', ''))}]")
        resolved_facts = result.get("resolved_facts")
        if resolved_facts:
            chunks.append("已解析事实：" + json.dumps(resolved_facts, ensure_ascii=False)[:1200])

        evidence_summary = result.get("evidence_summary") or result.get("evidence", {}).get("evidence_summary")
        if evidence_summary:
            chunks.append(str(evidence_summary)[:1200])
            continue

        for doc in result.get("documents", [])[:3]:
            payload = getattr(doc, "payload", {}) or {}
            text = payload.get("markdown_text", "") or ""
            if text:
                chunks.append(text[:800])
            proxy_descriptions = payload.get("proxy_descriptions", []) or []
            if proxy_descriptions:
                chunks.append("视觉代理描述：" + "\n".join([str(desc)[:400] for desc in proxy_descriptions[:2]]))
            image_paths = payload.get("extracted_crop_paths", []) or []
            if image_paths:
                chunks.append(f"关联图片：{', '.join(image_paths[:3])}")

    return "\n".join(chunks).strip() or "无可用前序证据"


def _collect_fact_values(value: Any) -> List[str]:
    """从事实抽取 JSON 中收集短值，避免把整段证据塞进检索 query。"""
    values = []
    if isinstance(value, dict):
        fact_value = value.get("value")
        if fact_value:
            values.append(str(fact_value).strip())
            return values
        for key, nested in value.items():
            if key in {"name", "evidence"}:
                continue
            values.extend(_collect_fact_values(nested))
    elif isinstance(value, list):
        for item in value:
            values.extend(_collect_fact_values(item))
    elif isinstance(value, str) and 0 < len(value.strip()) <= 80:
        values.append(value.strip())
    return values


def _format_dependency_query_hint(step_results: List[Dict[str, Any]], depends_on: List[str]) -> str:
    depends_on_set = set(depends_on)
    values = []

    for idx, result in enumerate(step_results, start=1):
        step = result.get("step", {})
        step_id = step.get("id") or f"step_{idx}"
        if depends_on_set and step_id not in depends_on_set:
            continue

        for value in _collect_fact_values(result.get("resolved_facts", {})):
            if value and value not in values:
                values.append(value)

        if values:
            continue

        evidence_summary = result.get("evidence_summary") or result.get("evidence", {}).get("evidence_summary", "")
        compact_summary = re.sub(r"\s+", " ", str(evidence_summary)).strip()
        if compact_summary and compact_summary != "无可用证据摘要":
            values.append(compact_summary[:120])

    return " ".join(values[:3]).strip()


def _merge_dependency_hint_into_query(query: str, query_hint: str, sub_question: str) -> str:
    query = (query or sub_question or "").strip()
    query_hint = (query_hint or "").strip()
    if not query_hint:
        return query

    hint_terms = [term for term in re.split(r"[\s,，;；|]+", query_hint) if term]
    if any(term in query for term in hint_terms):
        return query

    if "step_" in query or query == sub_question:
        query = re.sub(r"step_\d+\s*中确定的", "", sub_question).strip() or query
        query = re.sub(r"基于\s*step_\d+.*?[，,]", "", query).strip() or sub_question

    return f"{query_hint} {query}".strip()


def resolve_step_query_with_context(step: Dict[str, Any], state: GraphState) -> tuple[str, str]:
    """把带 step_x 占位的依赖子问题解析为当前可执行的检索查询。"""
    sub_question = step.get("sub_question", "")
    depends_on = step.get("depends_on", [])
    if not depends_on:
        return sub_question, ""

    step_results = state.get("step_results", [])
    prior_context = _format_prior_context(step_results, depends_on)
    if prior_context == "无可用前序证据":
        return sub_question, prior_context
    query_hint = _format_dependency_query_hint(step_results, depends_on)

    sys_prompt = resolver_prompts.get("system_prompt", "你是一个查询解析助手。")
    user_prompt = resolver_prompts.get("user_template", "{sub_question}").format(
        original_q=state.get("original_question", state.get("question", "")),
        sub_question=sub_question,
        purpose=step.get("purpose", ""),
        prior_context=prior_context,
    )

    try:
        resolved_query = _get_llm_client().generate(sys_prompt, user_prompt, temperature=0.0).strip()
        resolved_query = resolved_query.strip('"').strip("'")
        resolved_query = _merge_dependency_hint_into_query(resolved_query, query_hint, sub_question)
        return resolved_query or sub_question, prior_context
    except Exception as e:
        print(f"  ⚠️ 依赖查询解析失败，使用占位子问题检索: {e}")
        fallback_query = _merge_dependency_hint_into_query(sub_question, query_hint, sub_question)
        return fallback_query, prior_context


def resolve_step_query(step: Dict[str, Any], state: GraphState) -> str:
    resolved_query, _ = resolve_step_query_with_context(step, state)
    return resolved_query


def plan_query_node(state: GraphState) -> Dict:
    """节点：先规划子问题、多跳路径和每步检索模态。"""
    question = state["question"]
    original_q = state.get("original_question", question)
    text_only = state.get("text_only", False)

    print("--- 🧭 Agent 正在规划子问题与检索路径 ---")
    sys_prompt = planner_prompts.get("system_prompt", "你是一个检索规划助手。")
    user_prompt = planner_prompts.get("user_template", "{question}").format(
        question=question,
        original_q=original_q,
        text_only=text_only,
    )

    try:
        raw_response = _get_llm_client().generate(sys_prompt, user_prompt, temperature=0.0)
        plan = _normalize_plan(_extract_json_object(raw_response), question, text_only=text_only)
    except Exception as e:
        print(f"  ⚠️ 规划解析失败，降级为单步检索: {e}")
        plan = _default_plan(question, text_only=text_only)

    validation_result = _validate_or_repair_plan(original_q, plan, text_only=text_only)
    plan = validation_result["plan"]
    plan_validation = validation_result["validation"]

    for idx, step in enumerate(plan, start=1):
        print(f"  {idx}. [{step['mode']}] {step['sub_question']}")
    if plan_validation.get("issues"):
        print(f"  🧪 计划校验: {plan_validation}")

    return {
        "original_question": original_q,
        "retrieval_plan": plan,
        "plan_validation": plan_validation,
        "current_step_index": 0,
        "step_results": [],
        "step_attempts": [],
        "resolved_facts": {},
        "documents": [],
        "revision_number": 0,
        "step_should_retry": False,
        "agent_trace": [f"规划完成：{len(plan)} 个检索步骤"],
    }


def route_after_step(state: GraphState) -> str:
    """路由逻辑：当前步骤需要 fallback 则重试，否则进入下一步或生成。"""
    if state.get("step_should_retry", False):
        return "execute_step"

    plan = state.get("retrieval_plan", [])
    current_index = state.get("current_step_index", 0)
    if current_index >= len(plan):
        return "generate"
    return "execute_step"


def rewrite_query_node(state: GraphState) -> Dict:
    """兼容旧工作流：保留轻量 query fallback 能力。"""
    print("--- 🧠 触发 fallback：正在改写当前子问题 ---")
    question = state["question"]
    original_q = state.get("original_question", question)

    sys_prompt = router_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = router_prompts.get("user_template", "").format(original_q=original_q, question=question)

    try:
        better_query = _get_llm_client().generate(sys_prompt, user_prompt, temperature=0.5).strip()
        better_query = better_query.strip('"').strip("'")
        print(f"  ✨ fallback 搜索词: '{better_query}'")
    except Exception:
        better_query = question

    return {"question": better_query}
