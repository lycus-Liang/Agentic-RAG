#!/bin/bash

# 🌟 核心防雷：强制所有 HuggingFace 请求走国内镜像站！
export HF_ENDPOINT=https://hf-mirror.com

# 定义模型名称和端口 (如果以后换模型，只需要改这里)
MODEL_NAME="Qwen/Qwen3-8B"
PORT=6666
LOG_DIR="../log"
LOG_FILE="$LOG_DIR/vllm_server.log"

echo ""
echo "==================================================="
echo "🔧 激活 conda 虚拟环境：vllm"
echo "==================================================="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

echo "==========================================="
echo "  🧠 准备启动本地大模型 (vLLM) 后台服务"
echo "==========================================="

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 检查端口是否已经被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ 警告: 端口 $PORT 已经被占用！请先执行 stop_llm.sh 或检查是否有其他程序在使用该端口。"
    exit 1
fi

echo "⏳ 正在加载模型权重到 GPU，请耐心等待 (约需 1-3 分钟)..."
echo "📄 实时日志输出至: $LOG_FILE"

# 使用 nohup 在后台启动 vLLM，并将标准输出和错误输出都重定向到日志文件
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --served-model-name "$MODEL_NAME" \
    --port $PORT \
    --trust-remote-code \
    --gpu-memory-utilization 0.5 \
    --max-model-len 16384 \
    > "$LOG_FILE" 2>&1 &

# 获取后台进程的 PID
VLLM_PID=$!
echo "✅ vLLM 进程已在后台启动，PID: $VLLM_PID"
echo "👉 你可以使用命令查看实时启动进度: tail -f $LOG_FILE"
echo "==========================================="