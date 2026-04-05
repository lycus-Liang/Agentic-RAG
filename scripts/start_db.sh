#!/bin/bash
# start_db.sh - Qdrant 一键启动脚本

echo "🔌 正在唤醒 Qdrant 向量数据库..."

# 检查 6333 端口是否已经被占用 (说明已经启动了)
if lsof -Pi :6333 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Qdrant 已经在后台运行中，无需重复启动！"
    exit 0
fi

# 进入数据盘目录并启动
cd /root/autodl-tmp/qdrant
QDRANT__SERVICE__STATIC_CONTENT_DIR=$(pwd)/static nohup ./qdrant > qdrant.log 2>&1 &

echo "🚀 Qdrant 点火成功！"
echo "👉 请在浏览器访问 Web UI: http://localhost:6333/dashboard"