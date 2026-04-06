import os
import sys
import yaml
from typing import Dict
from src.retrieval.searcher import Searcher
from src.agent.state import GraphState
from src.utils.llm_client import LLMClient

# 全局加载配置与提示词
with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
    config_dict = yaml.safe_load(f)
with open("./configs/prompts.yaml", 'r', encoding='utf-8') as f:
    prompts_dict = yaml.safe_load(f)

searcher = Searcher(config_dict=config_dict)
llm_client = LLMClient(config_dict.get("online", {}).get("llm_api", {}))
gen_prompts = prompts_dict.get("agent_generator", {})

def retrieve_node(state: GraphState) -> Dict:
    question = state["question"]

    print(f"--- 🔍 检索执行: '{question}' ---")
    original_q = state.get("original_question", question)
    text_only = state.get("text_only", False)

    documents = searcher.search(question, text_only=text_only)
    
    return {"documents": documents, "question": question, "original_question": original_q}


def generate_node(state: GraphState) -> Dict:
    print("--- ✍️ 正在调用云端大模型生成最终图文答案 ---")
    original_question = state["original_question"]
    documents = state["documents"]

    # 获取从 UI 传进来的回调函数
    ui_cb = state.get("ui_stream_callback")
    
    context_text = ""
    image_paths = []
    
    for i, doc in enumerate(documents):
        payload = doc.payload
        context_text += f"\n[来源 {i+1} | 第 {payload.get('page_num')} 页]:\n{payload.get('markdown_text', '')}\n"
        crops = payload.get("extracted_crop_paths", [])
        for crop in crops:
            if os.path.exists(crop) and len(image_paths) < 3:
                image_paths.append(crop)

    sys_prompt = gen_prompts.get("system_prompt", "你是智能助手。")
    user_prompt = gen_prompts.get("user_template", "").format(context_text=context_text, question=original_question)
    
    # 打印一个华丽的分割线，准备迎接流式输出
    print("\n✨ 最终回答：\n")
    
    # 开启 stream=True，拿到生成器
    chunk_generator = llm_client.generate(
        system_prompt=sys_prompt, 
        user_prompt=user_prompt, 
        image_paths=image_paths, 
        is_final_answer=True,
        stream=True  # 激活流式传输！
    )

    output_text = ""
    # 实时捕获数据块并打印在屏幕上
    for chunk in chunk_generator:
        sys.stdout.write(chunk)
        sys.stdout.flush() # 强制刷新缓冲区，确保字是一个一个蹦出来的
        output_text += chunk # 拼接到全量字符串中，为了最后存入 State

        # 🌟 杀手锏：如果有网页端的回调函数，立刻把当前累积的文本发过去刷新页面！
        if ui_cb:
            ui_cb(output_text)

    print("\n") # 打印完毕后换个行

    # 将完整的结果送回给 LangGraph 的状态机
    return {"generation": output_text}