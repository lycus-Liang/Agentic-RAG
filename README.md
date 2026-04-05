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
./start_local_llm.sh          # 在后台启动 vLLM 大模型 API 服务 (默认端口 8000)
./stop_local_llm.sh           # 安全关闭大模型进程，释放珍贵的 GPU 显存

---

## 📦 二、 离线知识建库 (Ingestion & Build)
系统支持“单文件”与“文件夹批量”建库，并根据文件类型自动路由到对应的解析流水线。

**通用参数说明：**
* `--type`: 数据源类型，可选 `pdf` (默认，全模态解析) 或 `json` (纯文本极速解析)。
* `--clear`: [可选] 附加此参数将在建库前**清空**已有的 Qdrant 集合，防止数据重复。

### 📄 模式 A：PDF 全模态建库 (极其消耗算力与显存)
```bash
# 单个 PDF 文件建库
python main.py build --file ./data/raw/sample.pdf --type pdf

# 批量 PDF 文件夹建库（自动建立批次子目录，推荐加上 --clear 保持库纯洁）
python main.py build --dir ./data/raw/officeqa_test --type pdf --clear
```

### 📝 模式 B：JSON 纯文本极速建库 (物理隔离视觉模型)
```bash
# 单个 JSON 文件建库
python main.py build --file ./data/qa_corpus.json --type json

# 批量 JSON 文件夹建库
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