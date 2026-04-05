#!/bin/bash
# stop_db.sh - Qdrant 一键关闭脚本

echo "🛑 准备关闭 Qdrant 向量数据库..."

# 尝试优雅地发送终止信号
if pkill qdrant; then
    echo "✅ Qdrant 进程已成功终止，端口已释放！"
else
    echo "ℹ️ 未发现运行中的 Qdrant 进程，它可能已经被关闭了。"
fi