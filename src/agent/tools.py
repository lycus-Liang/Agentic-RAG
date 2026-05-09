import os
import sys
import yaml
from typing import Dict
from src.retrieval.searcher import Searcher
from src.agent.state import GraphState
from src.agent.router import resolve_step_query_with_context
from src.utils.llm_client import LLMClient

# 全局加载配置与提示词
with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", 'r', encoding='utf-8') as f:
    prompts_dict = yaml.safe_load(f)

_searcher = None
llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
gen_prompts = prompts_dict.get("agent_generator", {})


def get_searcher() -> Searcher:
    """延迟加载检索模型，避免 Streamlit 启动阶段因 HuggingFace 下载失败而崩溃。"""
    global _searcher
    if _searcher is None:
        _searcher = Searcher(config_dict=config_dict)
    return _searcher


def _format_retrieval_error(error: Exception) -> str:
    hf_endpoint = os.environ.get("HF_ENDPOINT", "未设置")
    return (
        f"检索模型初始化或检索失败: {error}\n"
        f"当前 HF_ENDPOINT={hf_endpoint}。如果是 HuggingFace SSL/下载问题，请切换可用镜像、取消 HF_ENDPOINT，"
        "或将 configs/config.yaml 中的模型名改为已下载的本地模型目录。"
    )


def _build_generation_fallback(original_question: str, context_text: str) -> str:
    evidence = (context_text or "").strip()
    if not evidence:
        return "未能生成最终答案，且当前没有可用检索证据。请检查知识库是否已建库、Qdrant 集合是否正确，以及 LLM 服务是否可用。"

    compact_evidence = evidence[:1600]
    return (
        "模型服务本次没有返回最终答案。以下是已检索到的证据摘要，供你继续排查或人工判断：\n\n"
        f"问题：{original_question}\n\n"
        f"{compact_evidence}"
    )


def retrieve_node(state: GraphState) -> Dict:
    question = state["question"]

    print(f"--- 🔍 检索执行: '{question}' ---")
    original_q = state.get("original_question", question)
    text_only = state.get("text_only", False)

    try:
        documents = get_searcher().search(question, text_only=text_only)
        retrieval_error = ""
    except Exception as e:
        retrieval_error = _format_retrieval_error(e)
        print(f"  ❌ {retrieval_error}")
        documents = []
    
    return {
        "documents": documents,
        "question": question,
        "original_question": original_q,
        "retrieval_error": retrieval_error,
    }


def execute_step_node(state: GraphState) -> Dict:
    """节点：执行当前计划步骤的检索，并按步骤选择检索模态。"""
    plan = state.get("retrieval_plan", [])
    step_index = state.get("current_step_index", 0)
    if step_index >= len(plan):
        return {"current_step_documents": [], "step_should_retry": False}

    step = dict(plan[step_index])
    if state.get("step_should_retry"):
        query = state.get("question") or step.get("sub_question") or state.get("original_question", "")
        dependency_context = state.get("dependency_context", "")
    else:
        query, dependency_context = resolve_step_query_with_context(step, state)
        query = query or state.get("question") or state.get("original_question", "")
    mode = step.get("mode", "hybrid")
    text_only = state.get("text_only", False)
    attempt_number = state.get("revision_number", 0) + 1

    print(f"--- 🔍 执行计划步骤 {step_index + 1}/{len(plan)} [{mode}]: '{query}' ---")

    try:
        documents = get_searcher().search(query, text_only=text_only, retrieval_mode=mode)
        retrieval_error = ""
    except Exception as e:
        retrieval_error = _format_retrieval_error(e)
        print(f"  ❌ {retrieval_error}")
        documents = []

    trace = list(state.get("agent_trace", []))
    if retrieval_error:
        trace.append(f"步骤 {step_index + 1} 检索失败：{retrieval_error}")
    else:
        trace.append(
            f"步骤 {step_index + 1} 检索完成：mode={mode}, query={query}, hits={len(documents)}"
        )

    return {
        "question": query,
        "current_step": step,
        "resolved_query": query,
        "dependency_context": dependency_context,
        "current_attempt_number": attempt_number,
        "current_step_documents": documents,
        "retrieval_error": retrieval_error,
        "current_step_attempt": {
            "step_index": step_index,
            "step_id": step.get("id", f"step_{step_index + 1}"),
            "attempt_number": attempt_number,
            "query": query,
            "mode": mode,
            "dependency_context": dependency_context,
            "hit_count": len(documents),
            "retrieval_error": retrieval_error,
        },
        "step_should_retry": False,
        "agent_trace": trace,
    }


def generate_node(state: GraphState) -> Dict:
    print("--- ✍️ 正在调用云端大模型生成最终图文答案 ---")
    original_question = state.get("original_question", state.get("question", ""))
    documents = state.get("documents", [])
    step_results = state.get("step_results", [])

    # 获取从 UI 传进来的回调函数
    ui_cb = state.get("ui_stream_callback")
    
    context_text = ""
    image_paths = []

    if step_results:
        for step_idx, result in enumerate(step_results, start=1):
            step = result.get("step", {})
            context_text += (
                f"\n[检索步骤 {step_idx} | 模态: {step.get('mode', 'hybrid')} | "
                f"子问题: {step.get('sub_question', '')}]\n"
                f"实际检索: {result.get('resolved_query', result.get('query', ''))}\n"
                f"目标: {step.get('purpose', '')}\n"
                f"证据状态: {'有效' if result.get('relevant') else '弱证据/无证据'}\n"
            )
            evidence_summary = result.get("evidence_summary", "")
            if evidence_summary:
                context_text += f"证据摘要:\n{evidence_summary}\n"

            for crop in result.get("image_paths", []):
                if os.path.exists(crop) and len(image_paths) < 3:
                    image_paths.append(crop)

            for i, doc in enumerate(result.get("documents", []), start=1):
                payload = doc.payload
                context_text += f"[来源 {i} | 第 {payload.get('page_num')} 页]:\n{payload.get('markdown_text', '')}\n"
                crops = payload.get("extracted_crop_paths", [])
                for crop in crops:
                    if os.path.exists(crop) and len(image_paths) < 3:
                        image_paths.append(crop)
    else:
        for i, doc in enumerate(documents):
            payload = doc.payload
            context_text += f"\n[来源 {i+1} | 第 {payload.get('page_num')} 页]:\n{payload.get('markdown_text', '')}\n"
            crops = payload.get("extracted_crop_paths", [])
            for crop in crops:
                if os.path.exists(crop) and len(image_paths) < 3:
                    image_paths.append(crop)

    sys_prompt = gen_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = gen_prompts.get("user_template", "").format(context_text=context_text, question=original_question)
    
    # 打印一个华丽的分割线，准备迎接流式输出
    print("\n✨ 最终回答：\n")
    
    output_text = ""
    try:
        # 开启 stream=True，拿到生成器
        chunk_generator = llm_client.generate(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths,
            is_final_answer=True,
            stream=True
        )

        # 实时捕获数据块并打印在屏幕上
        for chunk in chunk_generator:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            output_text += chunk

            if ui_cb:
                ui_cb(output_text)
    except Exception as e:
        print(f"⚠️ 流式生成失败 ({e})，将改用纯文本非流式兜底。")
        image_paths = []

    if not output_text.strip():
        print("⚠️ 流式生成未返回正文，自动切换为非流式生成兜底。")
        try:
            output_text = llm_client.generate(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                image_paths=image_paths,
                is_final_answer=True,
                stream=False
            )
            if not str(output_text).strip() and image_paths:
                print("⚠️ 带图非流式生成仍为空，改用纯文本非流式生成。")
                output_text = llm_client.generate(
                    system_prompt=sys_prompt,
                    user_prompt=user_prompt,
                    image_paths=[],
                    is_final_answer=True,
                    stream=False
                )
        except Exception as e:
            print(f"⚠️ 非流式生成失败 ({e})，最后尝试纯文本生成。")
            output_text = llm_client.generate(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                image_paths=[],
                is_final_answer=True,
                stream=False
            )
        if not str(output_text).strip():
            output_text = _build_generation_fallback(original_question, context_text)
        if ui_cb:
            ui_cb(output_text)

    print("\n") # 打印完毕后换个行

    # 将完整的结果送回给 LangGraph 的状态机
    return {"generation": output_text, "documents": documents}
