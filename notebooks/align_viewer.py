import os
import sys
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import tempfile

# 强行把项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.vision_worker import VisionWorker
from src.ingestion.text_worker import TextWorker
from src.ingestion.proxy_worker import ProxyWorker

# ==========================================
# ⚙️ 页面全局配置
# ==========================================
st.set_page_config(page_title="RAG 视觉对齐检验台", layout="wide", page_icon="🔬")

# ==========================================
# 🧠 模型单例缓存
# ==========================================
@st.cache_resource(show_spinner=False)
def load_workers():
    v_worker = VisionWorker()
    t_worker = TextWorker()
    p_worker = ProxyWorker()
    return v_worker, t_worker, p_worker

try:
    with st.spinner("⏳ 首次启动：正在将三大 AI 模型装载至 GPU 显存，请耐心等待 (约需 1-2 分钟)..."):
        vision_worker, text_worker, proxy_worker = load_workers()
    st.toast("✅ 大模型矩阵与视觉门卫已就绪！", icon="🟢")
except Exception as e:
    st.error(f"模型加载失败，请检查环境: {e}")
    st.stop()

# ==========================================
# 🎨 UI 布局与侧边栏
# ==========================================
st.title("🔬 Agentic Omni-Modal RAG 视觉对齐检验台")

with st.sidebar:
    st.header("📄 测试样本输入")
    uploaded_file = st.file_uploader("上传 PDF 文件进行抽查", type=["pdf"])
    
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        selected_page = st.slider("选择要抽查的页码", 1, total_pages, 1)
        run_btn = st.button("🚀 开始解析本页", type="primary", use_container_width=True)

# ==========================================
# 🚀 核心执行逻辑
# ==========================================
if uploaded_file and run_btn:
    page_index = selected_page - 1
    page = doc[page_index]
    
    # 渲染为高清图片供左侧展示
    pix = page.get_pixmap(dpi=150)
    img_path = os.path.join(temp_dir, f"page_{selected_page}.png")
    pix.save(img_path)
    
    # 提取单页 PDF 供 Omniparse 解析
    single_page_pdf_path = os.path.join(temp_dir, f"target_page_{selected_page}.pdf")
    single_doc = fitz.open()
    single_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
    single_doc.save(single_page_pdf_path)
    single_doc.close()
    
    col_left, col_right = st.columns([1, 1.2])
    
    # ------------------------------------------
    # 👈 左半部分：真值对比 (Ground Truth)
    # ------------------------------------------
    with col_left:
        st.subheader(f"📄 原始 PDF 第 {selected_page} 页")
        st.image(img_path, use_container_width=True)
        
    # ------------------------------------------
    # 👉 右半部分：三大检验 Tab
    # ------------------------------------------
    with col_right:
        st.subheader("⚙️ 解析与门卫侦测结果")
        tab1, tab2, tab3 = st.tabs(["📝 Markdown 文本", "🛡️ 视觉门卫 & VLM", "👁️ 向量探针"])
        
        with st.spinner("正在呼叫模型矩阵与安检通道..."):
            crop_save_dir = os.path.join(temp_dir, "crops")
            
            # 1. 文本与抠图提取
            markdown_text, extracted_crops = text_worker.extract_data_with_omniparse(single_page_pdf_path, crop_save_dir)
            
            # 2. VLM 代理描述 (附带门卫)
            proxy_descriptions = []
            if extracted_crops:
                # 🌟 修复幂等性 Bug：在 UI 测试时，清理门卫的记忆，防止反复测试同一页时误判为重复
                proxy_worker.gatekeeper.seen_hashes.clear()
                
                # 🌟 传入新增的 page_num 参数
                proxy_descriptions = proxy_worker.generate_proxy_batch(
                    extracted_crops, 
                    batch_size=1, 
                    page_num=selected_page
                )
                
            # 3. 视觉与文本向量提取
            vision_embeddings = vision_worker.process_image_batch([img_path], batch_size=1)
            text_embedding_sample = None
            if markdown_text.strip():
                test_embed = text_worker.embed_model.encode([markdown_text[:200]], return_dense=True)
                text_embedding_sample = test_embed['dense_vecs'][0]

        # 【Tab 1：文本对齐】
        with tab1:
            if markdown_text:
                st.success("✅ Markdown 提取成功")
                with st.expander("查看渲染效果", expanded=True):
                    st.markdown(markdown_text)
            else:
                st.warning("⚠️ 该页未提取出任何文本。")

        # 【Tab 2：门卫监控与 VLM 结果】
        with tab2:
            if extracted_crops:
                # 统计门卫拦截数据
                passed_count = sum(1 for d in proxy_descriptions if d != "")
                filtered_count = len(extracted_crops) - passed_count
                
                # 🌟 新增：监控仪表盘
                st.markdown("### 🚦 门卫拦截统计")
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("📦 总抠图数", len(extracted_crops))
                m_col2.metric("✅ 门卫放行", passed_count)
                m_col3.metric("❌ 门卫拦截", filtered_count)
                st.markdown("---")
                
                for idx, (crop_path, desc) in enumerate(zip(extracted_crops, proxy_descriptions)):
                    sub_col1, sub_col2 = st.columns([1, 2])
                    with sub_col1:
                        st.image(crop_path, caption=f"Crop {idx+1}")
                    with sub_col2:
                        # 🌟 新增：动态渲染拦截状态
                        if desc != "":
                            st.success("🟢 **状态: 门卫放行 (精肉)**")
                            st.markdown("**VLM Proxy 翻译结果：**")
                            st.info(desc)
                        else:
                            st.error("🔴 **状态: 被门卫拦截 (垃圾图)**")
                            st.caption("*(触发物理规则或感知哈希去重，已跳过大模型推理，节省算力)*")
                    st.markdown("---")
            else:
                st.info("ℹ️ 本页未检测到独立的复杂图片或图表。")

        # 【Tab 3：维度探测】
        with tab3:
            st.success("✅ 模型前向传播计算完毕")
            if vision_embeddings:
                import numpy as np
                shape = np.array(vision_embeddings[0]).shape
                st.code(f"ColPali Vision Tensor Shape: {shape}", language="python")