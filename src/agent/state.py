from typing import List, TypedDict, Annotated, Any
import operator
from qdrant_client.http.models import ScoredPoint

class GraphState(TypedDict):
    """
    Agent 运行时的全局状态存储 (它的“短时记忆”)
    """
    question: str                  # 用户的当前提问 (可能会被 Rewrite 节点修改)
    original_question: str         # 用户的原始提问 (存档，供最终生成时参考)
    documents: List[ScoredPoint]   # 检索到的相关文档节点
    generation: str                # 最终大模型生成的回答
    revision_number: int           # 反思重试的次数，防止死循环
    graded_relevant: bool          # 当前检索文档是否及格
    ui_stream_callback: Any        # 专门为 UI 流式输出准备的回调钩子
    text_only: bool                # 应对纯文本检索