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

# ==========================================
# 🎨 页面基础配置
# ==========================================
st.set_page_config(page_title="Omni-Modal RAG", page_icon="🧠", layout="wide")
st.title("🧠 Omni-Modal Agentic RAG")
st.markdown("支持**图文多模态检索**与**自主反思重写**的工业级大模型引擎。")

# ==========================================
# 🤖 核心引擎初始化 (缓存机制，避免每次点击重新加载)
# ==========================================
@st.cache_resource
def init_agent():
    return build_agentic_rag()

app = init_agent()

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
            "ui_stream_callback": stream_updater
        }
        final_answer = ""
        source_images = []
        
        try:
            for output in app.stream(inputs):
                for key, value in output.items():
                    if key == "plan_query":
                        plan = value.get("retrieval_plan", [])
                        validation = value.get("plan_validation", {})
                        status.write(f"🧭 动作: 已规划 {len(plan)} 个检索步骤。")
                        if validation.get("repaired"):
                            status.write(f"  计划已自动修复：{', '.join(validation.get('issues', [])[:3])}")
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
                    elif key == "generate":
                        status.write("✍️ 动作: 正在调用 VLM 生成图文融合答案...")
                        final_answer = value.get("generation", "")
                        
                        docs = value.get("documents", [])
                        for doc in docs:
                            crops = doc.payload.get("extracted_crop_paths", [])
                            for crop in crops:
                                if crop not in source_images and len(source_images) < 3:
                                    source_images.append(crop)

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
                        
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "images": source_images
            })

        except Exception as e:
            status.update(label="❌ 发生异常", state="error")
            st.error(f"Agent 运行出错: {str(e)}")
