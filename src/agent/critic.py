import yaml
from typing import Dict
from src.agent.state import GraphState
from src.utils.llm_client import LLMClient

with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", 'r', encoding='utf-8') as f:
    prompts_dict = yaml.safe_load(f)

critic_prompts = prompts_dict.get("agent_critic", {})
llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))

def grade_documents_node(state: GraphState) -> Dict:
    """节点：评估文档相关性 (LLM-as-a-judge)"""
    print("--- ⚖️ 考官正在评估文档 ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        print("  ❌ 没搜到任何东西。")
        return {"graded_relevant": False}

    context_text = "\n\n".join([doc.payload.get("markdown_text", "") for doc in documents])
    
    sys_prompt = critic_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = critic_prompts.get("user_template", "").format(context_text=context_text, question=question)
    
    try:
        score = llm_client.generate(sys_prompt, user_prompt, temperature=0.0).strip().lower()
    except Exception as e:
        print(f"  ⚠️ 考官打盹了 ({e})，直接放行。")
        score = "yes"

    if "yes" in score:
        print("  ✅ 考官裁定：相关！准许生成。")
        return {"graded_relevant": True}
    else:
        print("  ❌ 考官裁定：完全不沾边！打回去重搜。")
        rev_num = state.get("revision_number", 0) + 1
        return {"graded_relevant": False, "revision_number": rev_num}  