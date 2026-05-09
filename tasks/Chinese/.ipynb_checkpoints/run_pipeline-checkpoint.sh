#!/bin/bash

# 设置遇到错误立即停止执行
set -e 

# 强制指定 OpenMP 线程数，防止 libgomp 报错 (非常重要，上个问题中刚修复的)
export OMP_NUM_THREADS=4

# ==========================================
# 🌟 核心防雷设计：动态路径解析
# ==========================================
# 1. 动态获取脚本文件当前所在的绝对路径 (/root/rag/tasks/Chinese)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 2. 向上推算项目根目录的绝对路径 (/root/rag)
PROJECT_ROOT="$SCRIPT_DIR/../../"

# 3. 强制让脚本的执行环境跳转到项目根目录
cd "$PROJECT_ROOT"
echo "🌍 当前工作目录已切换至项目根目录: $(pwd)"

# ==========================================
# 📂 配置文件路径定义
# ==========================================
# 原始知识库语料文件 (读取这里的纯文本 JSONL，建库后数据将存入 Qdrant 数据库)
KNOWLEDGE_DIR="$PROJECT_ROOT/data/raw/xlj_fixed.jsonl"

# 测试用的问题文件路径 (根据你的要求精确指定)
TEST_JSONL="$PROJECT_ROOT/data/eval/test23.jsonl"               

# 压测结果输出路径 (统一放在 eval 目录下，保持项目整洁)
EVAL_OUTPUT="$PROJECT_ROOT/data/eval/eval_results_$(date +%Y%m%d_%H%M%S).json" 

echo "==========================================="
echo "  🚀 Omni-RAG 自动化建库与评测流水线启动 "
echo "==========================================="

echo -e "\n⏳ [步骤 1/3] 启动底层向量数据库 (Qdrant)..."
cd scripts
./start_db.sh
cd ..
# 等待几秒钟确保 Qdrant 完全启动
sleep 3 

echo -e "\n⏳ [步骤 2/3] 清理历史数据，并开始极速纯文本建库..."
if [ ! -f "$KNOWLEDGE_DIR" ]; then
    echo "❌ 错误: 找不到知识库原始语料文件 $KNOWLEDGE_DIR，请检查路径！"
    exit 1
fi
# 这里读取 KNOWLEDGE_DIR 下的 JSON，处理后由 Python 写入 Qdrant
# python main.py build --file "$KNOWLEDGE_DIR" --type jsonl --clear

echo -e "\n⏳ [步骤 3/3] 启动 LLM 自动化压测与判分..."
if [ ! -f "$TEST_JSONL" ]; then
    echo "❌ 错误: 找不到测试文件 $TEST_JSONL，请检查路径！"
    exit 1
fi
python -m tasks.Chinese.evaluate_json --test_file "$TEST_JSONL" --output_file "$EVAL_OUTPUT"

echo -e "\n✅ 流水线全部执行完毕！"
echo "👉 请打开 $EVAL_OUTPUT 查看最终大模型的答卷和打分情况。"