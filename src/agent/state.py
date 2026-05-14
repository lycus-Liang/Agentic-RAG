from typing import Any, Dict, List, TypedDict

try:
    from qdrant_client.http.models import ScoredPoint
except ImportError:
    ScoredPoint = Any


class GraphState(TypedDict, total=False):
    """
    Agent 运行时的全局状态存储 (它的“短时记忆”)
    """
    question: str                  # 用户的当前提问 (可能会被 Rewrite 节点修改)
    original_question: str         # 用户的原始提问 (存档，供最终生成时参考)
    documents: List[ScoredPoint]   # 检索到的相关文档节点
    generation: str                # 最终大模型生成的回答
    revision_number: int           # 单步 fallback 的次数，防止死循环
    graded_relevant: bool          # 当前检索文档是否及格
    step_should_retry: bool        # 当前步骤是否需要 fallback 后重试
    ui_stream_callback: Any        # 专门为 UI 流式输出准备的回调钩子
    text_only: bool                # 应对纯文本检索
    debug: bool                    # 是否输出/返回工具检索调试轨迹
    session_id: str                # 会话/用户标识，用于隔离 agent memory
    memory_context: str            # 检索到的历史策略经验，不能作为事实证据
    retrieved_memories: List[Dict[str, Any]]
    memory_trace: List[str]
    memory_record_id: str
    memory_action: str
    feedback_score: float
    retrieval_plan: List[Dict[str, Any]]  # 子问题、多跳路径与模态路由计划
    plan_validation: Dict[str, Any]
    current_step_index: int        # 当前执行到第几个计划步骤
    current_step: Dict[str, Any]   # 当前计划步骤快照
    resolved_query: str            # 当前步骤实际送入检索器的查询
    dependency_context: str        # 依赖步骤用于解析当前 query 的证据上下文
    current_attempt_number: int
    retry_reason: str
    current_step_documents: List[ScoredPoint]
    current_step_evidence: Dict[str, Any]
    current_step_attempt: Dict[str, Any]
    retrieval_error: str
    step_attempts: List[Dict[str, Any]]
    step_results: List[Dict[str, Any]]    # 每一步的检索证据与评估结果
    resolved_facts: Dict[str, Any]
    agent_trace: List[str]         # UI/日志可展示的轻量执行轨迹
    tool_trace: List[Dict[str, Any]]       # tool-call 检索轨迹
    tool_call_count: int
    tool_agent_stop_reason: str
