# 🚀 Agentic Omni-Modal RAG (全模态智能体检索增强生成系统)

本项目是一个企业级、支持多模态（PDF 复杂图文）与纯文本（JSON QA）双轨运行的 RAG 系统。底层融合了 BGE-M3 (稠密+稀疏双路) 与 ColPali/ColQwen2 (视觉多向量)，并通过大模型 Agent 实现智能查询重写、反思与动态路由。

## ⚠️ 环境与网络准备
由于项目依赖大量的 HuggingFace 开源模型与在线大模型 API，启动前请确保：
1. **已配置科学上网**，或配置了 HuggingFace 国内镜像源（如设置环境变量 `export HF_ENDPOINT=https://hf-mirror.com`）。
2. 在项目根目录的 `./configs/config.yaml` 和 `.env` 中正确配置了你的 LLM API Key 等信息。

---

## 🛠️ 一、 启动基础依赖服务 (Services)
在进行建库或检索前，必须先启动底层的向量数据库（Qdrant）和 OCR 解析引擎（Omniparse）。

进入脚本目录：
```bash
cd scripts/
```

### 1. 向量数据库 (Qdrant)
```bash
./start_db.sh       # 启动 Qdrant 服务
./stop_db.sh        # 关闭 Qdrant 服务
```

### 2. 多模态解析引擎 (Omniparse)
*注：仅处理纯文本 JSON 时无需开启此服务。*
```bash
./start_omniparse.sh    # 启动 Omniparse 容器
./stop_omniparse.sh     # 关闭 Omniparse 容器
```
*(操作完成后，请 `cd ..` 返回项目根目录执行后续 Python 命令)*

### 3. 本地私有化大模型服务 (vLLM) - [可选]
*注：如果你配置了云端 API（如 OpenAI、通义千问等），无需启动此项。如果你希望零成本使用本地显卡运行开源大模型（如 Qwen/Qwen3-8B），请先启动此服务。*
```bash
./start_local_llm.sh          # 在后台启动 vLLM 大模型 API 服务 (默认端口 8080)
./stop_local_llm.sh           # 安全关闭大模型进程，释放珍贵的 GPU 显存

---

## 📦 二、 离线知识建库 (Ingestion & Build)
系统采用了先进的**统一中间态与策略路由架构 (Strategy Registry)**，支持“单文件”与“文件夹批量”建库。无论是复杂的多模态 PDF，还是各类结构化纯文本（JSON, JSONL 等），系统都会自动路由到对应的解析流水线。

**通用参数说明：**
* `--type`: 指定数据源类型。当前内置支持 `pdf` (全模态图文解析)、`json` (标准问答语料)、`jsonl` (Alpaca 微调指令语料)。
* `--clear`: [可选] 附加此参数将在建库前**清空**已有的 Qdrant 集合，防止数据重复。

### 📄 模式 A：PDF 全模态建库 (深度解析图表与视觉特征)
```bash
# 单个 PDF 文件建库
python main.py build --file ./data/raw/sample.pdf --type pdf

# 批量 PDF 文件夹建库（推荐加上 --clear 保持库纯洁）
python main.py build --dir ./data/raw/officeqa_test --type pdf --clear
```

### 📝 模式 B：纯文本极速建库 (物理隔离视觉模型，支持多格式扩展)
纯文本模式下，系统将跳过耗时的视觉推理模型，极速提取双路文本向量。
```bash
# 单个 JSON（或其它类型，jsonl等） 文件建库
python main.py build --file ./data/qa_corpus.json --type json

# 批量 JSON（或其它类型，jsonl等） 文件夹建库
python main.py build --dir ./data/json_dataset --type json --clear
```

---

## 💬 三、 在线智能检索与对话 (Chat)

### 1. 全模态对话模式 (默认)
火力全开，同时激活 BGE-M3 (双路文本) 与 ColPali (视觉多向量) 进行 RRF 融合召回，适用于查阅复杂财报、带有图表的 PDF。
```bash
python main.py chat
```

### 2. 纯文本极速模式 (Text-Only)
**核心特性**：如果你当前检索的库是纯文本 JSON 建立的，**强烈建议加上 `--text-only` 参数**！该参数将从底层彻底隔离 ColPali 视觉模型的推理，瞬间极大降低检索延迟与显存占用。
```bash
python main.py chat --text-only
```

---

## 📊 四、 可视化 Web UI (Streamlit)
项目提供了两个极其强大的可视化面板，用于调试和演示。

```bash
# 1. 离线建库数据对齐可视化 (查看文本块、向量与裁剪图片的对应关系)
streamlit run notebooks/align_viewer.py --server.port 8501

# 2. 在线对话与检索结果可视化 (查看 Agent 脑回路、重写过程与召回溯源)
streamlit run notebooks/web_ui.py --server.port 6006
```

---

## 🧪 五、 开发与单元测试 (Dev & Test)
如果你修改了某个特定的 Ingestion 模块（如 `text_worker.py` 或 `vision_worker.py`），可以独立运行它进行测试，避免启动庞大的全链路：

```bash
# 替换为具体的模块名进行测试，例如：
python -m src.ingestion.text_worker
python -m src.ingestion.json_parser
```

---

## 🔌 六、API 微服务 (API Service)

为了方便将本项目接入现有的业务系统（如 Web 前端、微信小程序、企业内网），系统内置了基于 `FastAPI` 的高性能异步接口服务。

### 1. 启动 API 服务
在项目根目录下运行：
```bash
python api_server.py
```
服务默认将在 `http://0.0.0.0:6666` 启动。

### 2. 交互式 API 文档 (Swagger UI)
启动服务后，强烈建议在浏览器中访问：
👉 `http://<你的服务器IP>:6666/docs`

FastAPI 会自动生成可视化接口文档，可直接在网页上模拟发送请求、测试 RAG 检索和生成效果。

### 3. 核心 API 路由说明
系统对外暴露了两个核心接口，均支持 POST 请求：

#### /v1/retrieve (纯检索接口)
- 功能：仅执行多路并发检索，返回带有相关性打分的原始文档块，不经过大模型生成。
- 适用场景：你的系统已有独立的大模型，仅需要借用本项目的全模态检索能力。

#### /v1/chat (端到端问答接口)
- 功能：完整执行 Agentic RAG 流水线（查询重写 -> 检索 -> 提示词组装 -> 大模型思考 -> 流式/全量输出）。
- 适用场景：直接获取最终答案及溯源文档，开箱即用。

### 4. 调用示例 (Python)
第三方业务系统可以通过标准 HTTP 请求轻松接入：

```python
import requests

url = "http://localhost:6666/v1/chat"
payload = {
  "query": "请解释一下 BGE-M3 的双路召回原理？",
  "text_only": True,
  "session_id": "user_12345"  # 支持多用户并发会话隔离
}

response = requests.post(url, json=payload)
print(response.json())
```

---
