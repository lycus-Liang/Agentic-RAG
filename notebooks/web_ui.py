import streamlit as st
import os
import sys
from dotenv import load_dotenv

# 🌟 架构师铁律：必须最先加载环境变量，唤醒 API 和 LangSmith
load_dotenv(override=True)

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
            cols = st.columns(len(msg["images"]))
            for idx, img_path in enumerate(msg["images"]):
                if os.path.exists(img_path):
                    cols[idx].image(img_path, caption=f"来源图片 {idx+1}", use_column_width=True)

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
        def stream_updater(accumulated_text):
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
                    if key == "retrieve":
                        status.write("🔍 动作: 正在多路召回向量数据库...")
                    elif key == "grade_documents":
                        status.write("⚖️ 动作: 考官正在审核文档有效性...")
                    elif key == "rewrite_query":
                        status.write(f"🧠 动作: 触发反思！将搜索词重写为 👉 `{value.get('question', '')}`")
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
            
            # 4. 生成结束后，去掉那个闪烁的光标，渲染最终整洁的文本
            answer_placeholder.markdown(final_answer)
            
            if source_images:
                st.markdown("---")
                st.markdown("**🖼️ 参考视觉资料：**")
                cols = st.columns(len(source_images))
                for idx, img_path in enumerate(source_images):
                    if os.path.exists(img_path):
                        cols[idx].image(img_path, use_column_width=True)
                        
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "images": source_images
            })

        except Exception as e:
            status.update(label="❌ 发生异常", state="error")
            st.error(f"Agent 运行出错: {str(e)}")