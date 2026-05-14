import yaml
from langgraph.graph import StateGraph, END
from src.agent.state import GraphState
from src.agent.tool_call_agent import tool_call_agent_node
from src.agent.tools import generate_node, retrieve_node
from src.agent.router import plan_query_node
from src.agent.memory import retrieve_memory_node, update_memory_node

def build_agentic_rag():
    with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
        use_agentic_loop = yaml.safe_load(f).get("online", {}).get("workflow", {}).get("use_agentic_loop", True)

    workflow = StateGraph(GraphState)

    workflow.add_node("generate", generate_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("update_memory", update_memory_node)

    if use_agentic_loop:
        # 开启 Plan + Tool-Calling Agentic RAG：先规划，再通过原生 tool call 执行检索。
        print("🧠 规划 + Tool-Calling Agentic RAG 工作流已开启...")
        workflow.add_node("plan_query", plan_query_node)
        workflow.add_node("tool_call_agent", tool_call_agent_node)

        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "plan_query")
        workflow.add_edge("plan_query", "tool_call_agent")
        workflow.add_edge("tool_call_agent", "generate")
    else:
        # 🌟 关闭 Agent 模式：传统 RAG 直线打法
        print("⚡ 极速直线 RAG 模式已开启 (考官与反思已关闭)...")
        workflow.add_node("retrieve", retrieve_node)
        workflow.set_entry_point("retrieve_memory")
        workflow.add_edge("retrieve_memory", "retrieve")
        workflow.add_edge("retrieve", "generate")

    workflow.add_edge("generate", "update_memory")
    workflow.add_edge("update_memory", END)
    
    return workflow.compile()
