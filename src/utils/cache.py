import os
import pickle
from src.utils.logger import logger

def run_with_cache(cache_filepath: str, func, *args, **kwargs):
    """
    通用断点续传拦截器
    如果缓存文件存在，直接加载跳过；如果不存在，执行函数并保存缓存。
    """
    if os.path.exists(cache_filepath):
        logger.info(f"♻️ 触发断点续传！发现阶段缓存，直接秒级加载: {os.path.basename(cache_filepath)}")
        with open(cache_filepath, 'rb') as f:
            return pickle.load(f)
    
    # 如果没缓存，执行真正的耗时逻辑
    result = func(*args, **kwargs)
    
    # 跑完之后，立刻存档！
    with open(cache_filepath, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"💾 阶段结果已存档至: {os.path.basename(cache_filepath)}")
    
    return result