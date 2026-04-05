import logging
import os

path = "/root/rag/log/rag_pipeline.log"

def setup_logger():
    # 创建全局唯一的 Logger 实例
    logger = logging.getLogger("Agentic-RAG")
    logger.setLevel(logging.INFO)

    # 避免重复添加 Handler（多文件 import 时常见坑点）
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
        
        # 1. 终端输出
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 2. 文件输出 (追加模式)
        file_handler = logging.FileHandler(path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# 暴露出实例化好的 logger
logger = setup_logger()