import os
import json
import yaml
from src.retrieval.searcher import Searcher

def load_config(filepath: str) -> dict:
    """从外部加载全局 YAML 配置"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 找不到配置文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_golden_dataset(filepath: str) -> list:
    """从外部 JSON 文件动态加载评估集"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 找不到测试集文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate():
    # ==========================================
    # 1. 在最外层加载配置 (全局神经中枢)
    # ==========================================
    config_path = "./configs/config.yaml"
    global_config = load_config(config_path)
    
    # ==========================================
    # 2. 依赖注入：将配置字典传给 Searcher
    # ==========================================
    print("🚀 正在装载检索引擎与模型矩阵...")
    searcher = Searcher(config_dict=global_config)
    
    # 打印当前生效的检索策略，方便调试
    strategies = global_config.get("online", {}).get("retrieval_strategies", {})
    print(f"🎛️ 当前生效的检索策略: 稠密={strategies.get('use_dense')}, 稀疏={strategies.get('use_sparse')}, 视觉={strategies.get('use_vision')}")

    # ==========================================
    # 3. 加载测试集并开始评估
    # ==========================================
    dataset_path = "./data/eval/dataset.json"
    golden_dataset = load_golden_dataset(dataset_path)
    
    total_queries = len(golden_dataset)
    hits = 0
    mrr_sum = 0.0

    print(f"\n🔬 开始评估 {total_queries} 条黄金测试集...\n")

    # 获取检索的 top_k 值
    top_k = global_config.get("online", {}).get("top_k", {}).get("fr_top_k", 5)

    for idx, item in enumerate(golden_dataset):
        query = item["query"]
        expected_doc = item["expected_doc"]
        expected_page = item["expected_page"]

        print(f"[{idx+1}/{total_queries}] Q: {query}")
        
        # 发起多路融合检索
        retrieved_points = searcher.search(query)
        
        hit_found = False
        rank = 0
        
        for i, point in enumerate(retrieved_points):
            payload = point.payload
            if payload.get("source_doc") == expected_doc and payload.get("page_num") == expected_page:
                hit_found = True
                rank = i + 1
                break
                
        if hit_found:
            hits += 1
            mrr_sum += (1.0 / rank)
            print(f"  ✅ 命中! (Rank: {rank})")
        else:
            print(f"  ❌ 未命中! (期待: {expected_doc} 第{expected_page}页)")

    # ==========================================
    # 4. 输出量化报告
    # ==========================================
    hit_rate = (hits / total_queries) * 100
    mrr = mrr_sum / total_queries

    print("\n" + "="*40)
    print(f"📊 检索评估报告 (Top-{top_k})")
    print("="*40)
    print(f"🎯 Hit Rate (命中率): {hit_rate:.2f}%")
    print(f"🥇 MRR (平均倒数排名): {mrr:.4f}")

if __name__ == "__main__":
    evaluate()