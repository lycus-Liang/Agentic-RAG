import os
# from dotenv import load_dotenv

# # 寻找并加载项目根目录下的 .env 文件
# # 如果你以后是在 main.py 里启动整个项目，这段代码应该放在 main.py 的第一行
# load_dotenv() 

# # 验证一下是否成功注入
# print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# # ==========================================

import yaml
import torch
import imagehash
from PIL import Image
from typing import List, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from src.utils.device_manager import get_optimal_device

def is_blank_or_solid_color(img: Image.Image, threshold: float = 0.95) -> bool:
    """
    判断图片是否为空白或几乎纯色（无有效内容）
    :param img: PIL Image 对象
    :param threshold: 主色调占比阈值，超过则判定为纯色（默认 95%）
    :return: True 表示空白/纯色，需要过滤
    """
    # 缩小图片尺寸，加快计算
    img_small = img.resize((64, 64))
    # 统计所有像素的颜色出现次数
    pixel_counts = {}
    for pixel in img_small.getdata():
        pixel_counts[pixel] = pixel_counts.get(pixel, 0) + 1
    # 找到出现次数最多的颜色
    max_count = max(pixel_counts.values())
    total_pixels = 64 * 64
    # 主色调占比
    main_color_ratio = max_count / total_pixels
    return main_color_ratio > threshold


# ==========================================
# 🛡️ 视觉门卫：物理与防重过滤器
# ==========================================
class VisionGatekeeper:
    def __init__(self):
        # 用于记录已经处理过的图片哈希值，实现全局去重
        self.seen_hashes = set()

    def is_valid_image(self, image_path: str) -> Tuple[bool, str]:
        """
        核心过滤逻辑：判断图片是否有价值送给 VLM。
        返回 (是否有效, 无效原因)
        """
        try:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size

            # 1. 绝对尺寸与面积过滤 (杀掉小 Icon、标点符号)
            if width < 50 or height < 50:
                return False, f"尺寸过小 ({width}x{height})"
            if width * height < 4000:
                return False, f"面积过小 ({width * height} px²)"

            # 2. 空白/纯色图片过滤 (杀掉纯白、纯黑、纯灰等无内容图片)
            if is_blank_or_solid_color(img):
                return False, "空白或纯色图片"

            # 3. 极端长宽比过滤 (杀掉文档里的分割线、高亮条、页边距框)
            aspect_ratio = width / height
            if aspect_ratio > 8 or aspect_ratio < 0.125:
                return False, f"比例极端 (宽高比 {aspect_ratio:.2f})"

            # 4. 感知哈希去重 (杀掉每一页重复出现的公司 Logo、页眉页脚装饰)
            # pHash 会根据图片的视觉特征生成指纹，极其精准
            img_hash = str(imagehash.phash(img))
            if img_hash in self.seen_hashes:
                return False, f"重复图片 (pHash: {img_hash})"

            # 记录这页的 hash，放行！
            self.seen_hashes.add(img_hash)
            return True, "放行"
            
        except Exception as e:
            return False, f"读取失败 ({e})"

class ProxyWorker:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        初始化 VLM 代理工作站。
        默认使用 2B 版本，显存占用小（约 6GB），处理速度极快，足够胜任图片描述任务。
        如果你的显存充裕（>24GB），可以换成 Qwen2-VL-7B-Instruct。
        """
        self.device, self.dtype = get_optimal_device()
        print(f"🕵️‍♂️ 正在加载 Proxy VLM 模型到 {self.device} (精度: {self.dtype})...")
        
        # 加载模型
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        ).eval()
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_name)

        # 实例化门卫
        self.gatekeeper = VisionGatekeeper()

        # 从 YAML 配置文件动态加载 Prompt
        self.system_prompt = self._load_prompt()

        print("✅ Proxy VLM 模型、提示词与视觉门卫加载完毕！")

    def _load_prompt(self) -> str:
        """
        定位并解析项目根目录下的 configs/prompts.yaml
        """
        # 获取当前脚本的绝对路径，向上推两层找到项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, "../../configs/prompts.yaml")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                prompts_config = yaml.safe_load(f)
                # 安全地获取配置，如果文件里没写，给一个兜底的默认值
                return prompts_config.get("proxy_worker", {}).get(
                    "system_prompt", 
                    "请详细描述图片内容。" # 极端情况的兜底方案
                )
        except FileNotFoundError:
            print(f"⚠️ 警告: 未找到配置文件 {yaml_path}，将使用默认极简提示词。")
            return "请详细描述图片内容。"

    @torch.no_grad()
    def generate_proxy_batch(self, image_paths: List[str], batch_size: int = 4, page_num: int = None) -> List[str]:
        """
        批量生成图片的 Proxy 描述文本 (已启动视觉门卫过滤)。
        """
        # 动态生成日志前缀
        page_prefix = f"第 {page_num} 页的" if page_num is not None else "未知页码"
        
        # ==========================================
        # 🛡️ 第一步：视觉门卫进行垃圾图片过滤
        # ==========================================
        valid_paths = []
        filtered_count = 0
        
        for path in image_paths:
            is_valid, reason = self.gatekeeper.is_valid_image(path)
            if is_valid:
                valid_paths.append(path)
            else:
                filtered_count += 1
                print(f"    🛡️ 门卫拦截 [{os.path.basename(path)}]: {reason}")

        if filtered_count > 0:
            print(f"  -> 📉 {page_prefix} 共抠出 {len(image_paths)} 张图，门卫拦截了 {filtered_count} 张，放行 {len(valid_paths)} 张。")

        # 绝对保序占位符：不管拦截多少，返回的列表长度必须和入参一致
        final_descriptions = [""] * len(image_paths)
        if not valid_paths:
            return final_descriptions

        # ==========================================
        # 🧠 第二步：VLM 仅处理高价值图片
        # ==========================================
        valid_descriptions = []

        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i + batch_size]
            print(f"  -> 🖼️ 正在由 VLM 翻译 {page_prefix} 高价值插图: 第 {i+1} 到 {i+len(batch_paths)} 张 (共 {len(valid_paths)} 张)...")

            messages_batch = []
            for path in batch_paths:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": path},
                            {"type": "text", "text": self.system_prompt},
                        ],
                    }
                ]
                messages_batch.append(messages)

            # 准备输入数据
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages_batch
            ]
            
            image_inputs, video_inputs = process_vision_info(messages_batch)
            
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 批量推理生成
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            
            # 裁剪掉 prompt 部分，只保留生成的答案
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            valid_descriptions.extend(output_texts)
            
        # ==========================================
        # 🧩 第三步：严格对齐原始位置
        # ==========================================
        valid_idx = 0
        for i, path in enumerate(image_paths):
            if path in valid_paths:
                final_descriptions[i] = valid_descriptions[valid_idx]
                valid_idx += 1

        return final_descriptions

# ================= 测试入口 =================
if __name__ == "__main__":
    worker = ProxyWorker()
    # 假设这是 Omniparse 从 PDF 里抠出来的一张电路图或统计图
    test_images = ["./data/processed/RAG调研/extracted_image/crops/page_5_crop_0.jpg", "./data/processed/RAG调研/extracted_image/crops/page_5_crop_1.jpg"]
    
    import os
    if os.path.exists(test_images[0]):
        descriptions = worker.generate_proxy_batch(test_images, batch_size=2)
        for idx, desc in enumerate(descriptions):
            print(f"\n📊 图片 {idx+1} 的 Proxy 描述:\n{desc}\n{'-'*40}")
    else:
        print("准备好测试图片即可运行 VLM 代理翻译测试！")