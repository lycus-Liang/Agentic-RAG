#!/bin/bash

echo "==========================================="
echo "  🛑 准备关闭本地大模型 (vLLM) 服务"
echo "==========================================="

# 查找包含 vllm.entrypoints.openai.api_server 的进程 PID
VLLM_PIDS=$(pgrep -f "vllm.entrypoints.openai.api_server")

if [ -z "$VLLM_PIDS" ]; then
    echo "💡 没有检测到正在运行的 vLLM 服务。"
else
    echo "🔥 发现 vLLM 进程 PID: $VLLM_PIDS"
    echo "⏳ 正在安全终止进程释放显存..."
    
    # 遍历杀死所有相关进程
    for PID in $VLLM_PIDS; do
        kill -9 $PID
    done
    
    echo "✅ vLLM 服务已成功关闭！显存已释放。"
fi
echo "==========================================="