import json
import yaml
from typing import Dict
from src.agent.state import GraphState
from src.utils.llm_client import LLMClient

with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", 'r', encoding='utf-8') as f:
    prompts_dict = yaml.safe_load(f)

critic_prompts = prompts_dict.get("agent_critic", {})
step_prompts = prompts_dict.get("agent_step_assessor", {})
router_prompts = prompts_dict.get("agent_router", {})
fact_prompts = prompts_dict.get("agent_fact_extractor", {})
llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
MAX_RETRIES = config_dict.get("online", {}).get("workflow", {}).get("max_retries", 2)


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


def _build_step_evidence(documents, mode: str) -> Dict:
    image_paths = []
    proxy_descriptions = []
    text_snippets = []
    sources = []

    for doc in documents or []:
        payload = getattr(doc, "payload", {}) or {}
        source_doc = payload.get("source_doc", "unknown")
        page_num = payload.get("page_num", 0)
        sources.append({"source_doc": source_doc, "page_num": page_num})

        text = payload.get("markdown_text", "") or ""
        if text:
            text_snippets.append(text[:500])

        for desc in payload.get("proxy_descriptions", []) or []:
            if desc and desc not in proxy_descriptions:
                proxy_descriptions.append(str(desc)[:500])

        for path in payload.get("extracted_crop_paths", []) or []:
            if path and path not in image_paths:
                image_paths.append(path)

    summary_parts = []
    if text_snippets:
        summary_parts.append("文本证据：" + "\n".join(text_snippets[:3]))
    if proxy_descriptions:
        summary_parts.append("视觉代理描述：" + "\n".join(proxy_descriptions[:3]))
    if image_paths:
        summary_parts.append(f"关联图片数量：{len(image_paths)}")

    return {
        "mode": mode,
        "hit_count": len(documents or []),
        "image_paths": image_paths,
        "image_count": len(image_paths),
        "proxy_descriptions": proxy_descriptions,
        "sources": sources,
        "evidence_summary": "\n\n".join(summary_parts) or "无可用证据摘要",
    }


def _extract_resolved_facts(step: Dict, query: str, evidence: Dict) -> Dict:
    sys_prompt = fact_prompts.get("system_prompt", "你是证据事实抽取器。")
    user_template = fact_prompts.get("user_template", "")
    if not user_template:
        return {}

    user_prompt = user_template.format(
        step_json=json.dumps(step, ensure_ascii=False),
        query=query,
        evidence_summary=evidence.get("evidence_summary", ""),
    )

    try:
        raw_response = llm_client.generate(sys_prompt, user_prompt, temperature=0.0)
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        return json.loads(cleaned)
    except Exception:
        return {}


def grade_documents_node(state: GraphState) -> Dict:
    """节点：评估文档相关性 (LLM-as-a-judge)"""
    print("--- ⚖️ 考官正在评估文档 ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        print("  ❌ 没搜到任何东西。")
        return {"graded_relevant": False}

    context_text = "\n\n".join([doc.payload.get("markdown_text", "") for doc in documents])
    
    sys_prompt = critic_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = critic_prompts.get("user_template", "").format(context_text=context_text, question=question)
    
    try:
        score = llm_client.generate(sys_prompt, user_prompt, temperature=0.0).strip().lower()
    except Exception as e:
        print(f"  ⚠️ 考官打盹了 ({e})，直接放行。")
        score = "yes"

    if "yes" in score:
        print("  ✅ 考官裁定：相关！准许生成。")
        return {"graded_relevant": True}
    else:
        print("  ❌ 考官裁定：完全不沾边！打回去重搜。")
        rev_num = state.get("revision_number", 0) + 1
        return {"graded_relevant": False, "revision_number": rev_num}  


def assess_step_node(state: GraphState) -> Dict:
    """节点：评估当前子问题证据，决定进入下一跳或做一次轻量 fallback。"""
    print("--- ⚖️ Agent 正在评估当前检索步骤 ---")
    plan = state.get("retrieval_plan", [])
    step_index = state.get("current_step_index", 0)
    current_step = state.get("current_step", {})
    documents = state.get("current_step_documents", [])
    question = state.get("question") or current_step.get("sub_question", "")
    original_q = state.get("original_question", question)
    revision_number = state.get("revision_number", 0)
    retrieval_error = state.get("retrieval_error", "")

    if not current_step and step_index < len(plan):
        current_step = plan[step_index]

    evidence = _build_step_evidence(documents, current_step.get("mode", "hybrid"))
    if retrieval_error:
        print("  ❌ 当前步骤检索执行失败，跳过无意义重试并进入下一步/生成。")
        evidence["evidence_summary"] = retrieval_error

        current_attempt = dict(state.get("current_step_attempt", {}))
        current_attempt.update({
            "relevant": False,
            "retry_reason": "检索执行失败",
            "retrieval_error": retrieval_error,
            "evidence": evidence,
        })
        step_attempts = list(state.get("step_attempts", []))
        step_attempts.append(current_attempt)

        step_id = current_step.get("id", f"step_{step_index + 1}")
        step_results = list(state.get("step_results", []))
        step_results.append({
            "step": current_step,
            "query": question,
            "resolved_query": state.get("resolved_query", question),
            "dependency_context": state.get("dependency_context", ""),
            "attempts": [attempt for attempt in step_attempts if attempt.get("step_id") == step_id],
            "mode": current_step.get("mode", "hybrid"),
            "relevant": False,
            "documents": [],
            "evidence": evidence,
            "image_paths": [],
            "evidence_summary": retrieval_error,
            "resolved_facts": {},
        })

        trace = list(state.get("agent_trace", []))
        trace.append(f"步骤 {step_index + 1} 检索执行失败，已跳过重试")

        return {
            "step_results": step_results,
            "documents": state.get("documents", []),
            "current_step_evidence": evidence,
            "step_attempts": step_attempts,
            "current_step_index": step_index + 1,
            "revision_number": 0,
            "graded_relevant": False,
            "step_should_retry": False,
            "retry_reason": "检索执行失败",
            "retrieval_error": "",
            "agent_trace": trace,
        }

    relevant = False
    if documents:
        context_text = evidence["evidence_summary"]
        sys_prompt = step_prompts.get("system_prompt") or critic_prompts.get("system_prompt", "你是智能助手。")
        user_template = step_prompts.get("user_template") or critic_prompts.get("user_template", "")
        user_prompt = user_template.format(
            context_text=context_text,
            question=question,
            original_q=original_q,
            purpose=current_step.get("purpose", ""),
            mode=current_step.get("mode", "hybrid"),
            image_count=evidence["image_count"],
        )

        try:
            score = llm_client.generate(sys_prompt, user_prompt, temperature=0.0).strip().lower()
        except Exception as e:
            print(f"  ⚠️ 步骤评估失败 ({e})，保守放行当前证据。")
            score = "yes"
        relevant = "yes" in score

    if relevant:
        print("  ✅ 当前步骤证据有效，进入下一步。")
    elif revision_number < MAX_RETRIES:
        print("  ❌ 当前步骤证据不足，生成 fallback 搜索词后重试本步。")
        fallback_query = question
        sys_prompt = router_prompts.get("system_prompt", "你是智能助手。")
        user_prompt = router_prompts.get("user_template", "{question}").format(
            original_q=original_q,
            question=question,
        )
        try:
            candidate_query = llm_client.generate(sys_prompt, user_prompt, temperature=0.3).strip()
            candidate_query = candidate_query.strip('"').strip("'")
            if candidate_query:
                fallback_query = candidate_query
        except Exception:
            pass

        retry_reason = "当前步骤证据不足"
        current_attempt = dict(state.get("current_step_attempt", {}))
        current_attempt.update({
            "relevant": False,
            "retry_reason": retry_reason,
            "fallback_query": fallback_query,
            "evidence": evidence,
        })
        step_attempts = list(state.get("step_attempts", []))
        step_attempts.append(current_attempt)

        trace = list(state.get("agent_trace", []))
        trace.append(f"步骤 {step_index + 1} 第 {revision_number + 1} 次尝试证据不足，fallback 查询：{fallback_query}")

        return {
            "question": fallback_query,
            "current_step": current_step,
            "step_attempts": step_attempts,
            "revision_number": revision_number + 1,
            "graded_relevant": False,
            "step_should_retry": True,
            "retry_reason": retry_reason,
            "agent_trace": trace,
        }
    else:
        print("  ⚠️ 当前步骤达到重试上限，记录空/弱证据并继续。")

    step_results = list(state.get("step_results", []))
    current_attempt = dict(state.get("current_step_attempt", {}))
    current_attempt.update({
        "relevant": relevant,
        "retry_reason": "" if relevant else "达到重试上限后继续",
        "evidence": evidence,
    })
    step_attempts = list(state.get("step_attempts", []))
    step_attempts.append(current_attempt)

    step_id = current_step.get("id", f"step_{step_index + 1}")
    resolved_facts = dict(state.get("resolved_facts", {}))
    extracted_facts = _extract_resolved_facts(current_step, question, evidence)
    if extracted_facts:
        resolved_facts[step_id] = extracted_facts

    step_results.append({
        "step": current_step,
        "query": question,
        "resolved_query": state.get("resolved_query", question),
        "dependency_context": state.get("dependency_context", ""),
        "attempts": [attempt for attempt in step_attempts if attempt.get("step_id") == step_id],
        "mode": current_step.get("mode", "hybrid"),
        "relevant": relevant,
        "documents": documents,
        "evidence": evidence,
        "image_paths": evidence["image_paths"],
        "evidence_summary": evidence["evidence_summary"],
        "resolved_facts": extracted_facts,
    })

    trace = list(state.get("agent_trace", []))
    trace.append(f"步骤 {step_index + 1} 完成：{'有效证据' if relevant else '弱证据/无证据'}")

    return {
        "step_results": step_results,
        "documents": _dedupe_documents(state.get("documents", []), documents),
        "current_step_evidence": evidence,
        "step_attempts": step_attempts,
        "resolved_facts": resolved_facts,
        "current_step_index": step_index + 1,
        "revision_number": 0,
        "graded_relevant": relevant,
        "step_should_retry": False,
        "retry_reason": "",
        "agent_trace": trace,
    }
