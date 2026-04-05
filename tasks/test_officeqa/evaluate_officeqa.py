import pandas as pd
import time
from tqdm import tqdm
from src.agent.workflow import build_agentic_rag

def run_evaluation(test_csv="officeqa_test_targets.csv", output_csv="my_rag_results.csv"):
    print("🚀 启动 OfficeQA 地狱级自动化压测...")
    
    # 1. 加载题库
    df = pd.read_csv(test_csv)
    
    # 2. 唤醒你的 Agent
    app = build_agentic_rag()
    
    results = []
    
    # 3. 遍历每一道题
    for index, row in tqdm(df.iterrows(), total=len(df), desc="🧠 压测进度"):
        question = row['question']
        ground_truth = row['answer']
        uid = row['uid']
        
        start_time = time.time()
        
        try:
            # 🌟 这里的 invoke 不会触发流式打印，适合后台跑批
            # 注意：如果之前 state 结构有改动，这里传入对应的 inputs 即可
            inputs = {"question": question}
            final_state = app.invoke(inputs)
            
            # 提取大模型的最终回答
            model_answer = final_state.get("generation", "")
            # 提取大模型重写后的 Query (如果有的话，用于事后分析)
            rewritten_query = final_state.get("question", question)
            
        except Exception as e:
            model_answer = f"ERROR: {str(e)}"
            rewritten_query = "N/A"
            
        latency = time.time() - start_time
        
        # 记录战况
        results.append({
            "uid": uid,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "rewritten_query": rewritten_query,
            "latency_seconds": round(latency, 2)
        })
        
        # 实时存盘，防止中途断网或显存 OOM 导致数据丢失
        pd.DataFrame(results).to_csv(output_csv, index=False)

    print(f"\n🎉 压测完毕！所有结果已保存至 {output_csv}")

if __name__ == "__main__":
    run_evaluation()