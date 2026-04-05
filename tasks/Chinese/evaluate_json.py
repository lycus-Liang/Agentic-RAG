import json
import argparse
import time
import os
from tqdm import tqdm

# 🌟 架构师铁律：在导入任何底层大模型或 RAG 库之前，先加载环境变量！
from dotenv import load_dotenv
load_dotenv(override=True)
print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# ==========================================

# 🌟 1. 核心导入与 [纯文本模式] 物理隔离
from src.agent.tools import searcher
from src.agent.workflow import build_agentic_rag
from src.utils.llm_client import LLMClient
import yaml

# 强制关闭视觉检索，极大提升批处理速度和节省显存
searcher.strategies["use_vision"] = False
if hasattr(searcher.encoder, 'use_vision'):
    searcher.encoder.use_vision = False

def load_config():
    with open("./configs/config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def llm_judge(question: str, ground_truth: str, model_answer: str, llm_client: LLMClient) -> dict:
    """
    LLM-as-a-Judge: 让大模型化身无情的“高考阅卷老师”
    """
    if not ground_truth.strip():
        return {"score": 0, "reason": "无标准答案，无法判分"}

    sys_prompt = "你是一位极其严谨的语文高考阅卷专家。请根据提供的【标准答案】，客观评判【考生回答】的正确性。"
    user_prompt = f"""
    【考试题目】:
    {question}
    
    【标准答案】:
    {ground_truth}
    
    【考生回答】:
    {model_answer}
    
    【判分规则】:
    1. 如果是选择题，只要考生的回答中明确包含了标准答案的选项字母，即给 100 分。
    2. 如果是简答题，请判断考生的回答是否命中了标准答案的核心得分点。全中 100 分，部分命中给 20-80 分，完全不沾边给 0 分。
    3. 请严格以 JSON 格式输出，不要包含任何其他废话。格式如下：
    {{"score": 整数分数, "reason": "你的判卷理由简述"}}
    """
    try:
        # 要求 LLM 强制返回 JSON
        response = llm_client.generate(system_prompt=sys_prompt, user_prompt=user_prompt)
        # 简单清理可能带有的 markdown 标记
        clean_json = response.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_json)
        return result
    except Exception as e:
        return {"score": 0, "reason": f"判卷大模型解析失败: {e}"}

def run_evaluation(test_file: str, output_file: str):
    print(f"🚀 启动 JSONL 自动化压测流水线...\n测试文件: {test_file}")
    
    config = load_config()
    llm_client = LLMClient(config.get("online", {}).get("llm_api", {}))
    app = build_agentic_rag()
    
    # 1. 加载 JSONL 测试数据
    dataset = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    results = []
    total_score = 0
    scored_count = 0

    # 2. 遍历测试集，带进度条
    for i, data in enumerate(tqdm(dataset, desc="🧠 RAG 答卷进度")):
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        ground_truth = data.get("output", "")
        
        # 拼装最终的 Query
        full_query = f"{instruction}\n{input_text}".strip()
        
        start_time = time.time()
        try:
            # 独立 thread_id，防止上下文记忆互相污染
            thread_config = {"configurable": {"thread_id": f"eval_json_{i}"}}
            final_state = app.invoke({"question": full_query, "text_only": True}, config=thread_config)
            
            model_answer = final_state.get("generation", "")
        except Exception as e:
            model_answer = f"ERROR: 系统崩溃 {str(e)}"
            
        latency = round(time.time() - start_time, 2)
        
        # 3. 自动化判卷 (仅当 output 不为空时触发)
        score_info = {"score": "N/A", "reason": "未提供标准答案"}
        if ground_truth.strip():
            score_info = llm_judge(full_query, ground_truth, model_answer, llm_client)
            total_score += score_info.get("score", 0)
            scored_count += 1
            
        # 4. 记录结果
        result_entry = {
            "id": i,
            "instruction": instruction,
            "input": input_text,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "latency_seconds": latency,
            "score": score_info.get("score"),
            "judge_reason": score_info.get("reason")
        }
        results.append(result_entry)
        
        # 实时写入，防止中途断电白跑
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 5. 打印测试报告
    print("\n" + "="*50)
    print("🎉 自动化压测完毕！")
    print(f"📄 成功处理 {len(results)} 道题目，结果已保存至: {output_file}")
    if scored_count > 0:
        avg_score = total_score / scored_count
        print(f"🏆 机器阅卷平均分: {avg_score:.2f} / 100")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="输入的测试集文件 (.jsonl)")
    parser.add_argument("--output_file", type=str, default="./data/eval_results.json", help="输出的评估结果文件 (.json)")
    args = parser.parse_args()
    
    run_evaluation(args.test_file, args.output_file)