import os
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.retrieval.query_encoder import QueryEncoder
from FlagEmbedding import FlagReranker

class Searcher:
    # 🌟 架构升级：通过参数 config_dict 注入配置，彻底与本地文件系统解耦！
    def __init__(self, config_dict: dict):
        # 提取 online 链路的配置
        self.online_config = config_dict.get("online", {})
        # 🎛️ 获取策略开关
        self.strategies = self.online_config.get("retrieval_strategies", {})
        
        # 🎯 获取 top_k 阈值配置
        top_k_config = self.online_config.get("top_k", {})
        self.cr_top_k_withReranker = top_k_config.get("cr_top_k_withReranker", 100)
        self.cr_top_k = top_k_config.get("cr_top_k", 10)
        self.fr_top_k = top_k_config.get("fr_top_k", 5)
        
        # 获取数据库配置
        db_config = self.online_config.get("database", {})
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 6333)
        self.collection_name = db_config.get("collection_name", "omni_rag_docs")
        
        print(f"🔌 连接 Qdrant 数据库 ({host}:{port}) | 目标集合: {self.collection_name}")
        self.client = QdrantClient(host=host, port=port)
        
        # 将配置继续向下传递给 Encoder
        self.encoder = QueryEncoder(self.online_config)

        # 初始化 Reranker
        self.use_reranker = self.strategies.get("use_reranker", False)
        if self.use_reranker:
            reranker_model = self.online_config['models'].get('reranker', 'BAAI/bge-reranker-v2-m3')
            print(f"⚖️ 正在装载 Reranker 模型: {reranker_model}...")
            # 开启 FP16 节省显存并提速
            self.reranker = FlagReranker(reranker_model, use_fp16=True)
            print("✅ Reranker 就绪！")

    def search(self, query: str, top_k: int = None, text_only: bool = False):
        """
        受配置开关控制的多路并发召回 + Reranker 重排
        :param query: 搜索词
        :param top_k: 最终返回数量。若不传，则默认使用 config.yaml 中的 fr_top_k 精筛数量
        """
        # 决定最终输出数量
        final_top_k = top_k if top_k is not None else self.fr_top_k
        
        encoded = self.encoder.encode(query, text_only=text_only)
        prefetch_queries = []

        # 核心逻辑：根据是否开启 Reranker，决定粗排 (Recall) 捞取的数据量
        fetch_limit = self.cr_top_k_withReranker if self.use_reranker else self.cr_top_k
        prefetch_limit = fetch_limit * 2 # RRF 的底层召回池通常需要是粗排目标的 2 倍，确保融合质量
        
        # 路线 A：稠密检索开关
        if self.strategies.get("use_dense", True) and "bge_dense" in encoded:
            prefetch_queries.append(
                models.Prefetch(query=encoded["bge_dense"], using="bge_dense", limit=prefetch_limit)
            )
            
        # 路线 B：稀疏检索开关
        if self.strategies.get("use_sparse", True) and "bge_sparse" in encoded:
            prefetch_queries.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=encoded["bge_sparse"]["indices"],
                        values=encoded["bge_sparse"]["values"]
                    ),
                    using="bge_sparse", limit=prefetch_limit
                )
            )

        # 路线 C：视觉检索开关
        if not text_only and self.strategies.get("use_vision", False) and encoded.get("colpali_vision"):
            prefetch_queries.append(
                models.Prefetch(query=encoded["colpali_vision"], using="colpali_vision", limit=prefetch_limit)
            )

        # 防御性判断：如果全关了，直接报错
        if not prefetch_queries:
            raise ValueError("❌ 所有的检索策略都被关闭了，请检查 config.yaml！")

        # ==========================================
        # 1. 粗排阶段 (Recall): Qdrant RRF 融合
        # ==========================================
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit, 
            with_payload=True
        )

        points = results.points

        # ==========================================
        # 2. 精排阶段 (Rerank): 使用 Cross-Encoder 进行终极审判
        # ==========================================
        if self.use_reranker and points:
            sentence_pairs = []
            for point in points:
                # 从 payload 中拿回建库时存入的 markdown_text
                doc_text = point.payload.get("markdown_text", "")
                sentence_pairs.append([query, doc_text])
            
            # 计算重排分数
            scores = self.reranker.compute_score(sentence_pairs, normalize=True)
            
            # 兼容处理单条数据时返回 float 的情况
            if isinstance(scores, float):
                scores = [scores]
                
            # 将新分数注入并重新排序
            for i, point in enumerate(points):
                point.score = scores[i] 
                
            points.sort(key=lambda x: x.score, reverse=True)
            
        # ==========================================
        # 3. 统一输出截断
        # ==========================================
        # 无论是否走过精排，我们都截断到 final_top_k (即 config 中的 fr_top_k)
        return points[:final_top_k]