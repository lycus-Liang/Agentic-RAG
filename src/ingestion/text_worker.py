import os
# from dotenv import load_dotenv

# # 寻找并加载项目根目录下的 .env 文件
# # 如果你以后是在 main.py 里启动整个项目，这段代码应该放在 main.py 的第一行
# load_dotenv() 

# # 验证一下是否成功注入
# print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# # ==========================================

import requests
import json
from typing import Dict, Any, Tuple, List
import concurrent.futures
import base64
from pathlib import Path

import requests
from FlagEmbedding import BGEM3FlagModel
from src.utils.device_manager import get_optimal_device
from src.utils.logger import logger
from src.utils.retry_utils import retry_with_backoff

class TextWorker:
    def __init__(self, omniparse_url: str = "http://localhost:8000/parse_document/pdf", embed_model_name: str = "BAAI/bge-m3"):
        self.omniparse_url = omniparse_url

        self.device, self.dtype = get_optimal_device()
        
        print(f"📝 正在加载 {embed_model_name} 文本模型到 {self.device}...")
        
        # BGE-M3 的 use_fp16 逻辑：如果是 cpu 就设为 False，GPU 设为 True
        use_fp16 = (self.device != "cpu")
        
        self.embed_model = BGEM3FlagModel(
            embed_model_name, 
            use_fp16=use_fp16 
        )
        # 强制将模型推到对应的设备
        self.embed_model.model.to(self.device)
        print("✅ 文本向量模型加载完毕！")

    @retry_with_backoff(max_retries=3, initial_delay=3)
    def extract_data_with_omniparse(self, pdf_path: str, crop_save_dir: str) -> Tuple[str, List[str]]:
        """
        调用 Omniparse 服务，同时榨干文本和局部图片！
        返回: (Markdown 文本, 抠出的局部图片路径列表)
        """
        with open(pdf_path, 'rb') as f:
            files = {'file': (pdf_path, f, 'application/pdf')}
            response = requests.post(self.omniparse_url, files=files, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            markdown_text = result.get("text", "")
            
            # 默认给个空列表，防止 None
            images_data = result.get("images") or [] 
            saved_image_paths = []
            
            if images_data:
                os.makedirs(crop_save_dir, exist_ok=True)
                items_to_process = []
                
                # 🛡️ 战术动作一：探明敌情 (判断数据结构)
                if isinstance(images_data, dict):
                    # 如果是字典 {"img_name": "base64"}
                    items_to_process = images_data.items()
                elif isinstance(images_data, list):
                    # 如果是列表，挨个检查里面的元素
                    for idx, item in enumerate(images_data):
                        if isinstance(item, str):
                            # 🌟 修正：只用 crop_idx 标记它是本页的第几个抠图
                            items_to_process.append((f"crop_{idx}.jpg", item))
                        elif isinstance(item, dict):
                            # 列表里是字典 [{"name": "xxx", "base64": "xxx"}]
                            b64 = item.get('base64', item.get('content', item.get('image', '')))
                            # 🌟 修正：同样去掉带有误导性的 page 字眼
                            name = item.get('name', f"crop_{idx}.jpg")
                            if b64:
                                items_to_process.append((name, b64))
                
                # 🛡️ 战术动作二：统一处理保存
                for img_name, base64_data in items_to_process:
                    if not base64_data:
                        continue
                        
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                        
                    # 核心汇流：
                    # Path(pdf_path).stem 会提取出真实的来源，比如 "report_page_12"
                    # 加上 img_name，最终名字变成 -> "report_page_12_crop_0.jpg"
                    # 这样无论是哪一页的哪一张图，血统清清楚楚，绝对不会覆盖！
                    save_path = os.path.join(crop_save_dir, f"{Path(pdf_path).stem}_{img_name}")
                    
                    try:
                        with open(save_path, "wb") as img_file:
                            img_file.write(base64.b64decode(base64_data))
                        saved_image_paths.append(save_path)
                    except Exception as e:
                        print(f"⚠️ 图片 {img_name} 解码/保存失败: {e}")
                        
            return markdown_text, saved_image_paths
        else:
            error_msg = f"Omniparse 响应异常，状态码: {response.status_code}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    # def process_pdf_batch(self, pdf_paths: List[str], output_base_dir: str, max_threads: int = 8, batch_size: int = 8) -> List[dict]:
    #     """
    #     [全面升级版] 批量处理，并打包所有资产！
    #     返回的不再是简单的元组，而是一个结构化的情报字典。
    #     """
    #     # 为这批文件创建一个统一的局部图片存放地
    #     crop_dir = os.path.join(output_base_dir, "crops")
        
    #     print(f"📝 正在使用 {max_threads} 个线程并发请求 Omniparse (提取图文)...")
        
    #     # 1. 并发获取 Text 和 局部 Images
    #     # 🌟 高级技巧：预先分配好固定长度的数组，确保结果绝对保序
    #     parsed_results = [None] * len(pdf_paths)
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    #         # 创建 Future 到 原始索引(idx) 的映射字典
    #         future_to_idx = {
    #             executor.submit(self.extract_data_with_omniparse, path, crop_dir): i 
    #             for i, path in enumerate(pdf_paths)
    #         }
            
    #         completed = 0
    #         # 使用 as_completed 保证有任务一完成立马更新进度
    #         for future in concurrent.futures.as_completed(future_to_idx):
    #             # 获取这个 future 原本属于第几页
    #             original_idx = future_to_idx[future]
                
    #             try:
    #                 # 把结果精准填入预先分配的坑位里，彻底杜绝乱序！
    #                 parsed_results[original_idx] = future.result()
    #             except Exception as e:
    #                 print(f"❌ 第 {original_idx + 1} 页解析出现致命异常: {e}")
    #                 # 给一个兜底空数据，防止后续崩溃
    #                 parsed_results[original_idx] = ("", []) 
                
    #             completed += 1
    #             # 🌟 打印极其舒适的实时进度！
    #             print(f"  -> Omniparse 解析进度: {completed}/{len(pdf_paths)} 页完成 🚀")

    #     # 剥离出所有的文本，准备做向量化
    #     markdown_texts = [res[0] for res in parsed_results]
        
    #     print(f"🧠 正在将 {len(markdown_texts)} 段文本批量送入 BGE-M3 计算向量...")
        
    #     # 2. 批量语义降维 (省略了部分重复代码，核心逻辑同上一版)
    #     final_payloads = []
    #     for i in range(0, len(markdown_texts), batch_size):
    #         batch_texts = markdown_texts[i:i+batch_size]
    #         valid_texts = [t if t.strip() else "empty" for t in batch_texts]
            
    #         embeddings = self.embed_model.encode(
    #             valid_texts, return_dense=True, return_sparse=True, batch_size=batch_size
    #         )
            
    #         for j in range(len(batch_texts)):
    #             # 这里我们把所有东西打包成一个字典，这就是未来存入 Qdrant 的 Payload 雏形！
    #             payload = {
    #                 "markdown": batch_texts[j],
    #                 "dense_vec": embeddings['dense_vecs'][j].tolist(),
    #                 "sparse_vec": {str(k): float(v) for k, v in embeddings['lexical_weights'][j].items()},
    #                 # 🌟 关键点：把这页抠出来的小图路径也记下来！
    #                 "extracted_crops": parsed_results[i+j][1] 
    #             }
    #             final_payloads.append(payload)
                
    #     return final_payloads

    def process_pdf_batch(self, pdf_paths: List[str], output_base_dir: str, batch_size: int = 8) -> List[dict]:
        """
        [高可用求稳版] 适配单线程 Omniparse。
        上半场：稳扎稳打串行提取图文（防 Timeout）。
        下半场：GPU 大 Batch 批量提向量。
        """
        crop_dir = os.path.join(output_base_dir, "crops")
        parsed_results = []
        
        logger.info(f"📝 开始串行调用 Omniparse 提取 {len(pdf_paths)} 页图文数据...")
        
        # ==========================================
        # 1. 串行获取 Text 和 局部 Images (极度稳定)
        # ==========================================
        for idx, path in enumerate(pdf_paths):
            try:
                # 稳稳地一页一页请求，绝不给 Omniparse 压力
                res = self.extract_data_with_omniparse(path, crop_dir)
                parsed_results.append(res)
                logger.info(f"  -> Omniparse 解析进度: {idx + 1}/{len(pdf_paths)} 页完成 🚀")
            except Exception as e:
                logger.error(f"❌ 第 {idx + 1} 页解析出现致命异常: {e}")
                # 给一个兜底空数据，保持索引对齐
                parsed_results.append(("", []))

        # 剥离出所有的文本
        markdown_texts = [res[0] for res in parsed_results]
        
        logger.info(f"🧠 正在将 {len(markdown_texts)} 段文本按 Batch={batch_size} 送入 BGE-M3 计算向量...")
        
        # ==========================================
        # 2. 批量语义降维 (充分压榨 GPU)
        # ==========================================
        final_payloads = []
        for i in range(0, len(markdown_texts), batch_size):
            batch_texts = markdown_texts[i:i+batch_size]
            valid_texts = [t if t.strip() else "empty" for t in batch_texts]
            
            # GPU 喜欢一次性吃进多个文本
            embeddings = self.embed_model.encode(
                valid_texts, return_dense=True, return_sparse=True, batch_size=batch_size
            )
            
            for j in range(len(batch_texts)):
                payload = {
                    "markdown": batch_texts[j],
                    "dense_vec": embeddings['dense_vecs'][j].tolist(),
                    "sparse_vec": {str(k): float(v) for k, v in embeddings['lexical_weights'][j].items()},
                    "extracted_crops": parsed_results[i+j][1] 
                }
                final_payloads.append(payload)
                
        return final_payloads

    def process_pdf(self, pdf_path: str, output_base_dir: str) -> dict:
        """
        处理单页 PDF（与 batch 版本逻辑完全统一）：
        1. Omniparse 提取 Markdown + 图片
        2. 保存图片到 crops 文件夹
        3. BGE-M3 生成 dense / sparse 向量
        返回: 结构化字典（markdown, dense_vec, sparse_vec, extracted_crops）
        """
        # 1. 创建图片保存目录
        crop_dir = os.path.join(output_base_dir, "crops")
        
        # 2. 调用 Omniparse 提取【文本 + 图片】
        markdown_text, saved_image_paths = self.extract_data_with_omniparse(pdf_path, crop_dir)
        
        # 3. 如果没有文本，返回空结构
        if not markdown_text.strip():
            return {
                "markdown": "",
                "dense_vec": [],
                "sparse_vec": {},
                "extracted_crops": []
            }

        # 4. 生成文本向量
        embeddings = self.embed_model.encode(
            [markdown_text], 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )
        
        dense_vec = embeddings['dense_vecs'][0].tolist()
        sparse_dict = {str(k): float(v) for k, v in embeddings['lexical_weights'][0].items()}
        
        # 5. 返回和 process_pdf_batch 一样的结构
        return {
            "markdown": markdown_text,
            "dense_vec": dense_vec,
            "sparse_vec": sparse_dict,
            "extracted_crops": saved_image_paths  # 👈 这就是提取出来的图片路径
        }

# ================= 测试入口 =================
if __name__ == "__main__":
    worker = TextWorker()
    # test_pdf = "./data/processed/普通高中教科书语文必修/pdfs/page_1.pdf"
    # output_dir = "./data/processed/普通高中教科书语文必修/extracted_image"  # 输出根目录

    test_pdf = "./data/processed/RAG调研/pdfs/page_5.pdf"
    output_dir = "./data/processed/RAG调研/extracted_image" 

    if os.path.exists(test_pdf):
        # 调用新版 process_pdf
        result = worker.process_pdf(test_pdf, output_dir)
        
        print(f"📝 提取 Markdown 长度: {len(result['markdown'])} 字符")
        print(f"📊 稠密向量维度: {len(result['dense_vec'])}")
        print(f"🧲 稀疏特征关键词数量: {len(result['sparse_vec'])}")
        print(f"🖼️ 提取图片数量: {len(result['extracted_crops'])}")
        
        # 打印保存的图片路径
        for img_path in result["extracted_crops"]:
            print(f"✅ 图片已保存: {img_path}")
    else:
        print(f"{test_pdf} 路径不存在")