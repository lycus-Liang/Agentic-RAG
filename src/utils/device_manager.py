import os
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeviceManager")

def get_optimal_device() -> tuple[str, torch.dtype]:
    """
    智能硬件探测器：自动在 CUDA, MUSA 和 CPU 之间切换，并分配合适的精度。
    支持通过环境变量 FORCE_DEVICE 强制指定设备（用于测试）。
    """
    # 1. 优先读取环境变量 (最高优先级，方便你强制测试)
    forced_device = os.environ.get("FORCE_DEVICE", "").lower()
    if forced_device in ["cuda", "musa", "cpu"]:
        logger.info(f"🔧 开发者强制指定使用设备: {forced_device}")
        if forced_device == "cpu":
            return "cpu", torch.float32

    # 2. 探测 NVIDIA CUDA
    if (not forced_device or forced_device == "cuda") and torch.cuda.is_available():
        logger.info(f"🟢 探测到 NVIDIA CUDA! 设备数: {torch.cuda.device_count()}")
        # 现代 N 卡推荐 bfloat16，老卡推荐 float16
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 3. 探测 摩尔线程 MUSA (使用安全动态导入，防止在 N 卡机器上报错)
    if not forced_device or forced_device == "musa":
        try:
            import torch_musa
            if torch.musa.is_available():
                logger.info(f"🟠 探测到 摩尔线程 MUSA! 设备数: {torch.musa.device_count()}")
                # MUSA 当前生态推荐 float16 以保证最大兼容性
                return "musa", torch.float16 
        except ImportError:
            # 如果没装 torch_musa，说明这是一台普通的机器，静默跳过即可
            pass

    # 4. 终极兜底：CPU
    logger.warning("⚪ 未探测到 GPU 加速器，已回退到 CPU 模式 (推理速度将非常慢)")
    return "cpu", torch.float32

# 测试入口
if __name__ == "__main__":
    device, dtype = get_optimal_device()
    print(f"最终分配方案 -> 设备: {device}, 精度: {dtype}")