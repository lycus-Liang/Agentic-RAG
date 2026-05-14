import os
# from dotenv import load_dotenv

# # 寻找并加载项目根目录下的 .env 文件
# # 如果你以后是在 main.py 里启动整个项目，这段代码应该放在 main.py 的第一行
# load_dotenv() 

# # 验证一下是否成功注入
# print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# # ==========================================

import yaml
import json
import torch
from typing import Any, Dict, Optional
from FlagEmbedding import BGEM3FlagModel
from src.utils.llm_client import LLMClient
from colpali_engine.models import ColQwen2, ColQwen2Processor

class QueryEncoder:
    def __init__(self, config_dict: dict):
        print("🧠 正在初始化查询编码器 (Query Encoder)...")
        models = config_dict.get("models", {})
        self.strategies = config_dict.get("retrieval_strategies", {})
        
        # 1. 加载 BGE-M3 (Text Dense & Sparse)
        text_model_name = models.get("text_encoder", "BAAI/bge-m3")
        print(f"📝 加载文本检索模型: {text_model_name}")
        self.bge_model = BGEM3FlagModel(text_model_name, use_fp16=True)
        
        # 2. 按需加载 ColPali/ColQwen 视觉检索探针
        vision_model_name = models.get("vision_encoder", "vidore/colqwen2-v0.1")
        self.vision_device = models.get("vision_device", "cuda")
        self.vision_model = None
        self.vision_processor = None

        if self.strategies.get("use_vision", False):
            print(f"👁️ 加载视觉检索模型: {vision_model_name}")
            self.vision_model = ColQwen2.from_pretrained(
                vision_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.vision_device
            ).eval()
            self.vision_processor = ColQwen2Processor.from_pretrained(vision_model_name)
        else:
            print("⚡ 视觉检索未启用，跳过 ColQwen2 模型加载。")
        
        # 3. 初始化 HyDE 引擎组件 (如果开启)
        self.use_hyde = self.strategies.get("use_hyde", False)
        if self.use_hyde:
            print(f" 加载 HyDE 引擎")
            self.llm_client = LLMClient(config_dict.get("llm_api", {}))
            self.hyde_prompts = self._load_hyde_prompts()
            self.cache_file = "./data/cache/hyde_cache.json"
            self.hyde_cache = self._load_cache()

        print("✅ Query Encoder 就绪！")

    def _load_hyde_prompts(self) -> dict:
        prompt_path = "./configs/prompts.yaml"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("hyde_engine", {})

    def _load_cache(self) -> dict:
        """加载本地 HyDE 缓存"""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """持久化保存缓存"""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.hyde_cache, f, ensure_ascii=False, indent=2)

    def _get_hyde_expansion(self, query: str) -> str:
        """获取 HyDE 扩展文本 (带缓存与静默降级)"""
        # 命中缓存，直接返回！(0 成本，0 延迟)
        if query in self.hyde_cache:
            print("  ⚡ [HyDE 缓存命中] 跳过 API 调用。")
            return self.hyde_cache[query]

        sys_prompt = self.hyde_prompts.get("system_prompt", "你是一个知识库专家。")
        user_prompt = self.hyde_prompts.get("user_template", "{query}").format(query=query)

        try:
            # 发起 API 请求
            hypothetical_doc = self.llm_client.generate(sys_prompt, user_prompt)
            print(f"  ✨ [HyDE 扩展] {hypothetical_doc[:50]}...")
            
            # 写入缓存并保存
            self.hyde_cache[query] = hypothetical_doc
            self._save_cache()
            
            return hypothetical_doc

        except Exception as e:
            # 🌟 Option A: 强容错降级。只打印警告，返回空字符串，不阻断核心检索流程
            print(f"  ⚠️ [HyDE API 异常] 无法生成假设性文档 ({e})。系统已静默降级，将仅使用原始 Query 进行检索。")
            return ""

    @torch.no_grad()
    def encode(
        self,
        query: str,
        text_only: bool = False,
        retrieval_mode: str = "hybrid",
        use_hyde_override: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """将自然语言转化为三路召回向量钥匙"""
        retrieval_mode = (retrieval_mode or "hybrid").lower()
        if retrieval_mode not in {"text", "vision", "hybrid"}:
            retrieval_mode = "hybrid"
        if text_only:
            retrieval_mode = "text"

        use_text_encoder = retrieval_mode in {"text", "hybrid"}
        use_vision_encoder = retrieval_mode in {"vision", "hybrid"}
        encoded = {"query_text": query}

        # ==========================================
        # 1. HyDE 拦截与处理
        # ==========================================
        search_query_for_text = query
        use_hyde = self.use_hyde if use_hyde_override is None else bool(use_hyde_override)
        if use_text_encoder and use_hyde:
            hyde_expansion = self._get_hyde_expansion(query)
            if hyde_expansion: # 如果生成成功或命中缓存
                search_query_for_text = f"{query}。{hyde_expansion}"

        # ==========================================
        # 2. BGE-M3 提取 Dense 与 Sparse
        # ==========================================
        if use_text_encoder:
            clean_query = str(search_query_for_text).encode('utf-8', 'ignore').decode('utf-8')
            bge_out = self.bge_model.encode([clean_query], return_dense=True, return_sparse=True)
            encoded["bge_dense"] = bge_out['dense_vecs'][0].tolist()

            lexical_weights = bge_out['lexical_weights'][0]
            encoded["bge_sparse"] = {
                "indices": [int(k) for k in lexical_weights.keys()],
                "values": list(lexical_weights.values())
            }
        
        # ==========================================
        # 3. ColPali/ColQwen 提取视觉多向量探针
        # ==========================================
        if use_vision_encoder and self.vision_model is not None and self.vision_processor is not None:
            # 极其重要：处理 Query 和处理 Image 的 API 是不同的！
            vision_inputs = self.vision_processor.process_queries([query]).to(self.vision_device)
            vision_embeddings = self.vision_model(**vision_inputs)
            # 将 3D Tensor 转换为 Qdrant 需要的 List[List[float]]
            encoded["colpali_vision"] = vision_embeddings[0].cpu().float().numpy().tolist()
        elif not use_vision_encoder:
            print("  ⚡ [模式降维] 纯文本模式开启，已物理跳过 ColQwen2 视觉向量提取。")
        else:
            print("  ⚠️ [视觉不可用] 视觉检索模型未加载，无法生成 ColPali 查询向量。")

        return encoded
