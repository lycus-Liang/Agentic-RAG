import streamlit as st
import os
import sys
from dotenv import load_dotenv

# 🌟 架构师铁律：必须最先加载环境变量，唤醒 API 和 LangSmith
_preset_hf_endpoint = os.environ.get("HF_ENDPOINT")
load_dotenv(override=True)
if _preset_hf_endpoint is not None:
    if _preset_hf_endpoint.strip():
        os.environ["HF_ENDPOINT"] = _preset_hf_endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)

# 强行把项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.workflow import build_agentic_rag
from src.agent.retrieval_tools import get_searcher
from src.agent.memory import apply_memory_feedback
from src.utils.image_quality import filter_informative_images

# ==========================================
# 🎨 页面基础配置
# ==========================================
st.set_page_config(page_title="Omni-Modal RAG", page_icon="🧠", layout="wide")
st.title("🧠 Omni-Modal Agentic RAG")
st.markdown("支持**图文多模态检索**与**自主反思重写**的工业级大模型引擎。")

with st.sidebar:
    show_retrieval_process = st.checkbox("显示检索过程", value=False)
    session_id = st.text_input("Session ID", value="default_user")


def render_tool_trace(tool_trace):
    if not tool_trace:
        st.caption("暂无工具检索记录。")
        return

    for idx, trace in enumerate(tool_trace, start=1):
        step_label = trace.get("step_id") or f"step_{idx}"
        st.markdown(
            f"{idx}. `{step_label}` -> `{trace.get('tool_name', '')}` | "
            f"planned=`{trace.get('planned_mode', '')}` | "
            f"actual=`{trace.get('actual_mode', trace.get('mode', ''))}` | "
            f"style=`{trace.get('search_style', 'auto')}->{trace.get('effective_search_style', trace.get('search_style', 'auto'))}` | "
            f"hits=`{trace.get('hit_count', 0)}`"
        )
        st.caption(
            f"planned query: {trace.get('planned_query', '')} | "
            f"actual query: {trace.get('actual_query', trace.get('query', ''))}"
        )
        strategies = trace.get("effective_strategies", {}) or {}
        if strategies:
            st.caption(
                "effective strategies: "
                + ", ".join([f"{key}={value}" for key, value in strategies.items()])
                + f" | routes={', '.join(trace.get('routes', []) or []) or 'none'}"
                + f" | fusion={trace.get('fusion', '') or 'n/a'}"
            )
        notes = trace.get("strategy_notes", []) or []
        if notes:
            st.caption("strategy notes: " + ", ".join(notes))


def render_memory_trace(memory_trace):
    if not memory_trace:
        st.caption("暂无 memory 记录。")
        return
    for item in memory_trace:
        st.caption(str(item))


def render_feedback_controls(memory_id, session_id, key_prefix):
    if not memory_id:
        return
    cols = st.columns([1, 1, 6])
    if cols[0].button("有帮助", key=f"{key_prefix}_up"):
        ok = apply_memory_feedback(memory_id, session_id=session_id, score=1.0)
        st.toast("反馈已记录" if ok else "反馈记录失败")
    if cols[1].button("没帮助", key=f"{key_prefix}_down"):
        ok = apply_memory_feedback(memory_id, session_id=session_id, score=-1.0)
        st.toast("反馈已记录" if ok else "反馈记录失败")

# ==========================================
# 🤖 核心引擎初始化 (缓存机制，避免每次点击重新加载)
# ==========================================
@st.cache_resource
def init_agent():
    app = build_agentic_rag()
    get_searcher()
    return app

try:
    with st.spinner("正在加载检索模型与 Agent 工作流，请稍候..."):
        app = init_agent()
    st.success("模型加载完成，可以开始提问。", icon="✅")
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# 初始化聊天历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 💬 渲染历史对话 (含图片)
# ==========================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果历史消息里存了图片，一并渲染出来
        if "images" in msg and msg["images"]:
            with st.expander(f"参考视觉资料 ({len(msg['images'])} 张)", expanded=False):
                cols = st.columns(min(len(msg["images"]), 3))
                for idx, img_path in enumerate(msg["images"]):
                    if os.path.exists(img_path):
                        cols[idx % len(cols)].image(img_path, caption=f"来源图片 {idx+1}", width=180)
        if show_retrieval_process and msg.get("tool_trace"):
            with st.expander(f"检索过程 ({len(msg['tool_trace'])} 次)", expanded=False):
                render_tool_trace(msg["tool_trace"])
        if show_retrieval_process and msg.get("memory_trace"):
            with st.expander("Agent Memory", expanded=False):
                render_memory_trace(msg["memory_trace"])
        if msg["role"] == "assistant" and msg.get("memory_id"):
            render_feedback_controls(msg["memory_id"], msg.get("session_id", "default_user"), f"history_{id(msg)}")

# ==========================================
# 🚀 核心对话流
# ==========================================
if prompt := st.chat_input("请输入您的问题"):
    
    # 1. 渲染并保存用户提问
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Agent 思考与回答阶段
    with st.chat_message("assistant"):
        # 1. 状态框 (Agent 脑回路)
        status = st.status("🤖 Agent 正在思考与检索...", expanded=True)

        # 2. 核心：在状态框下方创建一个空的“占位符”，专门用来流式写字
        answer_placeholder = st.empty()

        # 3. 定义回调函数：每次收到新文本，就加上个闪烁光标渲染到页面上
        streamed_answer = {"text": ""}

        def stream_updater(accumulated_text):
            streamed_answer["text"] = accumulated_text
            answer_placeholder.markdown(accumulated_text + " ▌")
        
        # 将回调函数塞进输入字典里，传给 LangGraph
        inputs = {
            "question": prompt, 
            "ui_stream_callback": stream_updater,
            "debug": show_retrieval_process,
            "session_id": session_id,
        }
        final_answer = ""
        source_images = []
        tool_trace = []
        memory_trace = []
        memory_id = ""
        
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "retrieve_memory":
                        memory_trace = value.get("memory_trace", [])
                        if show_retrieval_process:
                            status.write(f"🧠 动作: Agent Memory 检索完成。{', '.join(memory_trace)}")
                            memories = value.get("retrieved_memories", [])
                            for idx, memory in enumerate(memories, start=1):
                                status.write(
                                    f"  Memory {idx}: reward={memory.get('reward', 0):.2f} "
                                    f"{str(memory.get('profile', ''))[:80]}"
                                )
                    if key == "plan_query":
                        plan = value.get("retrieval_plan", [])
                        validation = value.get("plan_validation", {})
                        status.write(f"🧭 动作: 已规划 {len(plan)} 个检索步骤。")
                        if validation.get("repaired"):
                            repair_reasons = validation.get("pre_repair_issues", [])
                            status.write(
                                f"  初始计划触发自动修复，已修复为当前 {len(plan)} 步计划。"
                            )
                            if repair_reasons:
                                status.write(f"  修复原因：{', '.join(repair_reasons[:3])}")
                        final_issues = validation.get("final_issues", [])
                        if final_issues:
                            status.warning(f"  当前计划仍有风险：{', '.join(final_issues[:3])}")
                        for idx, step in enumerate(plan, 1):
                            deps = step.get("depends_on", [])
                            dep_text = f" | 依赖: {', '.join(deps)}" if deps else ""
                            status.write(f"  Step {idx}: `{step.get('mode', 'hybrid')}` {step.get('sub_question', '')}{dep_text}")
                    elif key == "execute_step":
                        step = value.get("current_step", {})
                        resolved_query = (
                            value.get("resolved_query")
                            or value.get("question")
                            or step.get("sub_question", "")
                        )
                        attempt_number = value.get("current_attempt_number", 1)
                        dependency_context = value.get("dependency_context", "")
                        status.write(
                            f"🔍 动作: Step `{step.get('id', '')}` / Attempt {attempt_number}，"
                            f"执行 `{step.get('mode', 'hybrid')}` 检索，"
                            f"实际查询 `{resolved_query}`，"
                            f"命中 {len(value.get('current_step_documents', []))} 条证据。"
                        )
                        if value.get("retrieval_error"):
                            status.error(value["retrieval_error"])
                        if dependency_context:
                            status.write(f"  依赖证据摘要: {dependency_context[:180]}...")
                    elif key == "assess_step":
                        if value.get("step_should_retry"):
                            attempts = value.get("step_attempts", [])
                            current_attempt = attempts[-1] if attempts else {}
                            status.write(
                                f"🧠 动作: Step `{current_attempt.get('step_id', '')}` / "
                                f"Attempt {current_attempt.get('attempt_number', '?')} 证据不足，"
                                f"原因: {value.get('retry_reason', '证据不足')}，"
                                f"下一次查询 `{value.get('question', '')}`"
                            )
                        else:
                            evidence = value.get("current_step_evidence", {})
                            attempts = value.get("step_attempts", [])
                            current_attempt = attempts[-1] if attempts else {}
                            status.write(
                                f"⚖️ 动作: Step `{current_attempt.get('step_id', '')}` / "
                                f"Attempt {current_attempt.get('attempt_number', '?')} 评估完成，"
                                f"图片 {evidence.get('image_count', 0)} 张，"
                                f"来源 {len(evidence.get('sources', []))} 条。"
                            )
                    elif key == "retrieve":
                        status.write("🔍 动作: 正在多路召回向量数据库...")
                    elif key == "tool_call_agent":
                        tool_trace = value.get("tool_trace", [])
                        if show_retrieval_process:
                            status.write(f"🧰 动作: 模型完成 {len(tool_trace)} 次工具检索。")
                            for idx, trace in enumerate(tool_trace, start=1):
                                status.write(
                                    f"  {trace.get('step_id', f'Step {idx}')}: "
                                    f"计划 `{trace.get('planned_mode', '')}` / "
                                    f"`{trace.get('planned_query', '')}`，"
                                    f"实际 `{trace.get('tool_name', '')}` / "
                                    f"`{trace.get('actual_query', trace.get('query', ''))}`，"
                                    f"风格 `{trace.get('search_style', 'auto')}->{trace.get('effective_search_style', trace.get('search_style', 'auto'))}`，"
                                    f"命中 {trace.get('hit_count', 0)} 条。"
                                )
                    elif key == "generate":
                        status.write("✍️ 动作: 正在调用 VLM 生成图文融合答案...")
                        final_answer = value.get("generation", "")
                        
                        docs = value.get("documents", [])
                        for doc in docs:
                            crops, _ = filter_informative_images(
                                doc.payload.get("extracted_crop_paths", []),
                                doc.payload.get("proxy_descriptions", []),
                            )
                            for crop in crops:
                                if crop not in source_images and len(source_images) < 3:
                                    source_images.append(crop)
                    elif key == "update_memory":
                        memory_trace = value.get("memory_trace", memory_trace)
                        memory_id = value.get("memory_record_id", "")
                        if show_retrieval_process:
                            status.write(
                                f"🧠 动作: Agent Memory 写入完成 "
                                f"`{value.get('memory_action', '')}` / `{memory_id}`"
                            )

            status.update(label="✅ Agent 思考完毕", state="complete", expanded=False)
            if not final_answer.strip():
                final_answer = streamed_answer["text"]
            
            # 4. 生成结束后，去掉那个闪烁的光标，渲染最终整洁的文本
            if final_answer.strip():
                answer_placeholder.markdown(final_answer)
            else:
                answer_placeholder.warning("本次检索已完成，但模型没有返回最终答案。请检查 LLM 服务日志或切换非视觉/纯文本配置重试。")
            
            if source_images:
                st.markdown("---")
                with st.expander(f"参考视觉资料 ({len(source_images)} 张)", expanded=False):
                    cols = st.columns(min(len(source_images), 3))
                    for idx, img_path in enumerate(source_images):
                        if os.path.exists(img_path):
                            cols[idx % len(cols)].image(img_path, width=180)

            if show_retrieval_process:
                with st.expander(f"检索过程 ({len(tool_trace)} 次)", expanded=False):
                    render_tool_trace(tool_trace)
                with st.expander("Agent Memory", expanded=False):
                    render_memory_trace(memory_trace)

            render_feedback_controls(memory_id, session_id, f"current_{memory_id or 'none'}")
                        
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "images": source_images,
                "tool_trace": tool_trace,
                "memory_trace": memory_trace,
                "memory_id": memory_id,
                "session_id": session_id,
            })

        except Exception as e:
            status.update(label="❌ 发生异常", state="error")
            st.error(f"Agent 运行出错: {str(e)}")
