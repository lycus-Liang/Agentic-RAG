import time
import logging
from functools import wraps

logger = logging.getLogger("OmniRAG")

def retry_with_backoff(max_retries=3, initial_delay=2, backoff_factor=2):
    """
    网络请求重试装饰器：遇到异常时自动等待并重试
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"❌ 连续 {max_retries} 次网络请求失败: {func.__name__}，放弃挣扎。报错: {e}")
                        raise e  # 最后一次依然失败，则把错误抛出去
                    
                    logger.warning(f"⚠️ 网络请求抖动 ({func.__name__}): {e}。将在 {delay} 秒后进行第 {attempt + 1} 次重试...")
                    time.sleep(delay)
                    delay *= backoff_factor  # 延迟时间翻倍 (2s -> 4s -> 8s)
        return wrapper
    return decorator