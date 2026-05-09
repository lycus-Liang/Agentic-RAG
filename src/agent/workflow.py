import yaml
from langgraph.graph import StateGraph, END
from src.agent.state import GraphState
from src.agent.tools import execute_step_node, generate_node, retrieve_node
from src.agent.critic import assess_step_node
from src.agent.router import plan_query_node, route_after_step

def build_agentic_rag():
    with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
        use_agentic_loop = yaml.safe_load(f).get("online", {}).get("workflow", {}).get("use_agentic_loop", True)

    workflow = StateGraph(GraphState)

    workflow.add_node("generate", generate_node)

    if use_agentic_loop:
        # 开启真正的 Agentic RAG：先规划，再按步骤检索、评估与综合。
        print("🧠 Agentic 规划式工作流已开启...")
        workflow.add_node("plan_query", plan_query_node)
        workflow.add_node("execute_step", execute_step_node)
        workflow.add_node("assess_step", assess_step_node)

        workflow.set_entry_point("plan_query")
        workflow.add_edge("plan_query", "execute_step")
        workflow.add_edge("execute_step", "assess_step")
        workflow.add_conditional_edges(
            "assess_step",
            route_after_step,
            {
                "execute_step": "execute_step",
                "generate": "generate"
            }
        )
    else:
        # 🌟 关闭 Agent 模式：传统 RAG 直线打法
        print("⚡ 极速直线 RAG 模式已开启 (考官与反思已关闭)...")
        workflow.add_node("retrieve", retrieve_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")

    workflow.add_edge("generate", END)
    
    return workflow.compile()
