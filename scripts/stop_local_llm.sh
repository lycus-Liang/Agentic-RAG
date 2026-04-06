#!/bin/bash

echo "🛑 正在执行 vLLM 清理..."

# 1. 击杀 vLLM API 服务器 (前端)
pkill -9 -f "vllm.entrypoints"

# 2. 🌟 核心杀招：专门击杀底层 C++ 引擎 (解决你的 2088 僵尸)
pkill -9 -f "VLLM::EngineCore"

# 3. 扫荡所有名字里带 vllm 的漏网之鱼
pkill -9 -f "vllm"

# 4. 释放 6666 端口（如果你用的是其他端口，请把 6666 改掉）
fuser -k -9 6666/tcp >/dev/null 2>&1

echo "✅ 显存已彻底释放！"