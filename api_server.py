import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# 🌟 架构师铁律：在导入任何底层大模型或 RAG 库之前，先加载环境变量！
from dotenv import load_dotenv
load_dotenv(override=True)
print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# ==========================================

# 1. 导入你现有的 RAG 核心组件
from src.agent.tools import searcher
from src.agent.workflow import build_agentic_rag

# 初始化 Agent 图状态机
agent_app = build_agentic_rag()

# 初始化 FastAPI 应用
app = FastAPI(
    title="Omni-Modal RAG API",
    description="企业级全模态检索增强生成服务接口",
    version="1.0.0"
)

# 配置跨域请求 (CORS)，允许网页前端直接调用你的 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境中建议把 * 换成具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📦 数据验证模型 (Pydantic Models)
# ==========================================
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="用户的搜索词或问题")
    top_k: Optional[int] = Field(None, description="期望返回的文档片段数量")
    text_only: bool = Field(False, description="是否开启纯文本极速模式 (关闭视觉检索)")
    retrieval_mode: str = Field("hybrid", description="检索模态：text、vision 或 hybrid")

class DocumentDTO(BaseModel):
    source_doc: str
    page_num: int
    markdown_text: str
    score: float
    images: List[str] = []

class RetrieveResponse(BaseModel):
    query: str
    documents: List[DocumentDTO]

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户的提问")
    text_only: bool = Field(False, description="是否开启纯文本极速模式")
    # 如果系统是多用户的，可以用 session_id 隔离记忆
    session_id: str = Field("default_user", description="用于隔离对话历史的会话ID") 

class ChatResponse(BaseModel):
    answer: str
    sources: List[DocumentDTO]

# ==========================================
# 🔌 接口 1：纯检索服务 (只捞数据，不生成)
# ==========================================
@app.post("/v1/retrieve", response_model=RetrieveResponse, summary="执行多路向量检索")
async def api_retrieve(request: RetrieveRequest):
    try:
        # 直接调用底层的 search 方法
        points = searcher.search(
            query=request.query, 
            top_k=request.top_k, 
            text_only=request.text_only,
            retrieval_mode=request.retrieval_mode
        )
        
        docs = []
        for p in points:
            payload = p.payload
            docs.append(DocumentDTO(
                source_doc=payload.get("source_doc", "unknown"),
                page_num=payload.get("page_num", 0),
                markdown_text=payload.get("markdown_text", ""),
                score=p.score,
                images=payload.get("extracted_crop_paths", [])
            ))
            
        return RetrieveResponse(query=request.query, documents=docs)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

# ==========================================
# 🔌 接口 2：端到端问答服务 (检索 + 大模型生成)
# ==========================================
@app.post("/v1/chat", response_model=ChatResponse, summary="执行端到端 RAG 问答")
async def api_chat(request: ChatRequest):
    try:
        # 利用传入的 session_id 作为图状态机的 thread_id，实现多用户并发不串线
        config = {"configurable": {"thread_id": request.session_id}}
        
        initial_state = {
            "question": request.query,
            "text_only": request.text_only
        }
        
        # 触发 Agent 工作流
        final_state = agent_app.invoke(initial_state, config=config)
        
        # 提取结果
        answer = final_state.get("generation", "")
        raw_docs = final_state.get("documents", [])
        
        docs = []
        for p in raw_docs:
            payload = p.payload
            docs.append(DocumentDTO(
                source_doc=payload.get("source_doc", "unknown"),
                page_num=payload.get("page_num", 0),
                markdown_text=payload.get("markdown_text", ""),
                score=p.score,
                images=payload.get("extracted_crop_paths", [])
            ))
            
        return ChatResponse(answer=answer, sources=docs)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答生成失败: {str(e)}")

# 主函数入口
if __name__ == "__main__":
    print("🚀 启动 Omni-RAG API 服务...")
    # host设为 0.0.0.0 允许外部网络访问，端口可自定义
    uvicorn.run(app, host="0.0.0.0", port=6666)
