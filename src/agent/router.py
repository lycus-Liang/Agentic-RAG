import yaml
from typing import Dict
from src.agent.state import GraphState
from src.utils.llm_client import LLMClient

with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", 'r', encoding='utf-8') as f:
    prompts_dict = yaml.safe_load(f)
rou_prompts = prompts_dict.get("agent_router", {})

llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
MAX_RETRIES = config_dict.get("online", {}).get("workflow", {}).get("max_retries", 2)

def route_after_grading(state: GraphState) -> str:
    """路由逻辑：及格就去生成，不及格就去重写"""
    if state.get("graded_relevant", False):
        return "generate"
    else:
        if state.get("revision_number", 0) > MAX_RETRIES:
            print("  ⚠️ 达到最大重试次数！只能强行生成了...")
            return "generate"
        return "rewrite_query"

def rewrite_query_node(state: GraphState) -> Dict:
    """节点：反思并重写用户的提问"""
    print("--- 🧠 触发反思：正在重写搜索词 ---")
    question = state["question"]
    original_q = state.get("original_question", question)
    
    sys_prompt = rou_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = rou_prompts.get("user_template", "").format(original_q=original_q, question=question)
    
    try:
        better_query = llm_client.generate(sys_prompt, user_prompt, temperature=0.5).strip()
        # 清理可能带有的引号
        better_query = better_query.strip('"').strip("'")
        print(f"  ✨ 新搜索词诞生: '{better_query}'")
    except Exception:
        better_query = question # 失败则兜底

    return {"question": better_query}