#!/bin/bash
# start_omniparse.sh - 全自动启动 OmniParse（VPN + conda + 切目录 + 启动服务）

echo "==================================================="
echo "⚠️  自动启动 VPN 加速（HuggingFace 模型依赖）"
echo "==================================================="
source /etc/network_turbo

echo ""
echo "==================================================="
echo "🔧 激活 conda 虚拟环境：omniparse-venv"
echo "==================================================="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omniparse-venv

echo ""
echo "==================================================="
echo "📂 进入 OmniParse 程序目录：/root/autodl-tmp/omniparse"
echo "==================================================="
cd /root/autodl-tmp/omniparse

echo ""
echo "==================================================="
echo "🔌 启动 OmniParse 解析服务..."
echo "==================================================="

# 检查端口是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ OmniParse 已经在运行！"
    exit 0
fi

python server.py --host 0.0.0.0 --port 8000 --documents