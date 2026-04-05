import os
# from dotenv import load_dotenv

# # 寻找并加载项目根目录下的 .env 文件
# # 如果你以后是在 main.py 里启动整个项目，这段代码应该放在 main.py 的第一行
# load_dotenv() 

# # 验证一下是否成功注入
# print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# # ==========================================

import torch
from PIL import Image
from typing import List
from colpali_engine.models import ColQwen2, ColQwen2Processor

# 🌟 引入我们写的设备管理器
from src.utils.device_manager import get_optimal_device

class VisionWorker:
    def __init__(self, model_name: str = "vidore/colqwen2-v0.1"):
        # 🌟 一行代码搞定设备和精度分配
        self.device, self.dtype = get_optimal_device()
        
        print(f"👁️ 正在加载 ColPali 视觉模型到 {self.device}...")
        
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)
        print("✅ 视觉模型加载完毕！")

    @torch.no_grad()
    def process_image(self, image_path: str) -> List[float]:
        """
        处理单张高清图片，返回视觉多向量（Multi-vector Embeddings）。
        """
        image = Image.open(image_path).convert("RGB")
        
        # 预处理图片并转为 tensor
        batch_images = self.processor.process_images([image]).to(self.device)
        
        # 提取特征
        embeddings = self.model(**batch_images)
        
        # 将 tensor 转为普通的 Python List，方便后续写入 Qdrant
        # 注意：ColPali 返回的是 multi-vector (比如 1024 块, 每块 128 维)
        # 具体入库时 Qdrant 的 multivector 配置需要与之对应
        embeddings_list = embeddings[0].cpu().float().numpy().tolist()
        
        return embeddings_list

    @torch.no_grad()
    def process_image_batch(self, image_paths: List[str], batch_size: int = 4) -> List[List[float]]:
        """
        工业级批量视觉推理：一次性吃进多张图片，榨干 GPU 算力。
        注意：batch_size 取决于你的显存大小，24G 显存通常可设为 4 或 8。
        """
        all_embeddings = []
        
        # 将总任务切分成多个小 batch
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"👁️ 正在批量提取视觉向量: {i+1} 到 {i+len(batch_paths)} 张...")
            
            # 同时打开多张图片
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # 批量预处理并送入 GPU
            batch_inputs = self.processor.process_images(images).to(self.device)
            
            # 批量推理！这就是提速 10 倍的核心！
            embeddings = self.model(**batch_inputs)
            
            # 将张量解包，存回普通列表
            for emb in embeddings:
                all_embeddings.append(emb.cpu().float().numpy().tolist())
                
        return all_embeddings

# ================= 测试入口 =================
if __name__ == "__main__":
    worker = VisionWorker()
    # 假设这是你刚才用 splitter 切出来的一张图
    test_img = "./data/processed/普通高中教科书语文必修/images/page_1.png"
    import os
    if os.path.exists(test_img):
        vec = worker.process_image(test_img)
        print(f"📊 成功提取视觉向量！向量块数量: {len(vec)}, 维度: {len(vec[0]) if len(vec)>0 else 0}")
    else:
        print(f"测试 img：{test_img} 不存在")