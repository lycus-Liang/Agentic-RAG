import yaml
from langgraph.graph import StateGraph, END
from src.agent.state import GraphState
from src.agent.tools import retrieve_node, generate_node
from src.agent.critic import grade_documents_node
from src.agent.router import route_after_grading, rewrite_query_node

def build_agentic_rag():
    with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
        use_agentic_loop = yaml.safe_load(f).get("online", {}).get("workflow", {}).get("use_agentic_loop", True)

    workflow = StateGraph(GraphState)

    # 基础节点一定存在
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")

    if use_agentic_loop:
        # 🌟 开启 Agent 模式：织入考官与反思节点，形成闭环
        print("🧠 Agentic 循环工作流已开启...")
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("rewrite_query", rewrite_query_node)

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            route_after_grading,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        workflow.add_edge("rewrite_query", "retrieve")
    else:
        # 🌟 关闭 Agent 模式：传统 RAG 直线打法
        print("⚡ 极速直线 RAG 模式已开启 (考官与反思已关闭)...")
        workflow.add_edge("retrieve", "generate")

    workflow.add_edge("generate", END)
    
    return workflow.compile()