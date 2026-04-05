import os
import uuid
import base64
from typing import Dict, Any, List
from qdrant_client.http import models

class PointBuilder:
    # @staticmethod
    # def encode_image_to_base64(image_path: str) -> str:
    #     """将物理图片转为 Base64，塞入 Payload 供前端展示"""
    #     if not image_path:
    #         return ""
    #     try:
    #         with open(image_path, "rb") as f:
    #             return base64.b64encode(f.read()).decode('utf-8')
    #     except Exception as e:
    #         print(f"⚠️ 图片转 Base64 失败 ({image_path}): {e}")
    #         return ""

    @staticmethod
    def build_point(
        source_doc: str, 
        page_num: int, 
        markdown_text: str, 
        dense_vec: list, 
        sparse_vec: dict, 
        vision_multivec: list, 
        crop_paths: list, 
        proxy_descriptions: list
    ) -> models.PointStruct:
        """
        组装多模态节点。拒绝 Base64，采用绝对路径存储！
        """
        # 1. 基于文档名和页码生成确定性的 UUID，防止重复入库
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_doc}_page_{page_num}"))
        
        # 🌟 防御性编程：将所有图片路径转化为服务器上的绝对路径
        abs_crop_paths = [os.path.abspath(p) for p in crop_paths]
        
        # 2. 组装行李箱 (Payload)
        payload = {
            "source_doc": source_doc,
            "page_num": page_num,
            "markdown_text": markdown_text,
            "proxy_descriptions": proxy_descriptions, # VLM 的幻觉翻译也存进去
            "extracted_crop_paths": abs_crop_paths # 存本地绝对路径
        }

        # 3. 组装三把锁 (Named Vectors)
        # Sparse Vector 需要把 dict 拆成 indices 和 values
        sparse_indices = [int(k) for k in sparse_vec.keys()]
        sparse_values = list(sparse_vec.values())

        # 4. 基础必备的文本双路向量
        vector_struct = {
            "bge_dense": dense_vec,
            # 稀疏向量需要用 Qdrant 专属的结构体包起来并挂载
            "bge_sparse": models.SparseVector(
                indices=sparse_indices,
                values=sparse_values
            )
        }
        # 5. 动态判断：只有当视觉向量真实存在时，才把这个 key 加入字典！
        if vision_multivec is not None:
            vectors["colpali_vision"] = vision_multivec

        # 6. 返回完整结构体
        return models.PointStruct(
            id=point_id,
            vector=vector_struct,
            payload=payload
        )