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
        self.last_search_metadata = {}

    def _resolve_effective_strategies(self, search_style: str) -> dict:
        """Map agent-facing search_style to per-call backend strategy switches."""
        requested_style = (search_style or "auto").lower()
        if requested_style not in {"auto", "exact", "semantic", "expanded"}:
            requested_style = "auto"

        use_dense = bool(self.strategies.get("use_dense", True))
        use_sparse = bool(self.strategies.get("use_sparse", True))
        use_hyde = bool(self.strategies.get("use_hyde", False))
        effective_style = requested_style
        notes = []

        if requested_style == "exact":
            use_hyde = False
            if use_sparse:
                use_dense = False
            elif use_dense:
                notes.append("exact_sparse_disabled_fallback_to_dense")
            else:
                notes.append("exact_no_text_route_enabled")
        elif requested_style == "semantic":
            use_hyde = False
            if not use_dense and use_sparse:
                notes.append("semantic_dense_disabled_fallback_to_sparse")
        elif requested_style == "expanded":
            if not use_hyde:
                effective_style = "semantic"
                notes.append("expanded_hyde_disabled_by_config")

        return {
            "requested_search_style": requested_style,
            "effective_search_style": effective_style,
            "use_dense": use_dense,
            "use_sparse": use_sparse,
            "use_hyde": use_hyde,
            "use_vision": bool(self.strategies.get("use_vision", False)),
            "use_reranker": self.use_reranker,
            "notes": notes,
        }

    def search(
        self,
        query: str,
        top_k: int = None,
        text_only: bool = False,
        retrieval_mode: str = "hybrid",
        search_style: str = "auto",
    ):
        """
        受配置开关控制的多路并发召回 + Reranker 重排
        :param query: 搜索词
        :param top_k: 最终返回数量。若不传，则默认使用 config.yaml 中的 fr_top_k 精筛数量
        :param retrieval_mode: text / vision / hybrid，用于 Agentic RAG 的步骤级模态路由
        :param search_style: auto / exact / semantic / expanded，由 Agent 选择的检索风格
        """
        # 决定最终输出数量
        final_top_k = top_k if top_k is not None else self.fr_top_k
        effective = self._resolve_effective_strategies(search_style)

        retrieval_mode = (retrieval_mode or "hybrid").lower()
        if retrieval_mode not in {"text", "vision", "hybrid"}:
            retrieval_mode = "hybrid"

        vision_enabled = effective["use_vision"]
        if text_only:
            retrieval_mode = "text"
        elif retrieval_mode == "vision" and not vision_enabled:
            print("  ⚠️ [模态降级] vision 检索未启用，降级为 text 检索。")
            retrieval_mode = "text"

        encode_text_only = text_only or retrieval_mode == "text" or not vision_enabled
        encoded = self.encoder.encode(
            query,
            text_only=encode_text_only,
            retrieval_mode=retrieval_mode,
            use_hyde_override=effective["use_hyde"],
        )
        prefetch_queries = []
        routes = []

        # 核心逻辑：根据是否开启 Reranker，决定粗排 (Recall) 捞取的数据量
        fetch_limit = self.cr_top_k_withReranker if self.use_reranker else self.cr_top_k
        prefetch_limit = fetch_limit * 2 # RRF 的底层召回池通常需要是粗排目标的 2 倍，确保融合质量
        
        use_text_routes = retrieval_mode in {"text", "hybrid"}
        use_vision_route = retrieval_mode in {"vision", "hybrid"}

        # 路线 A：稠密检索开关
        if use_text_routes and effective["use_dense"] and "bge_dense" in encoded:
            prefetch_queries.append(
                models.Prefetch(query=encoded["bge_dense"], using="bge_dense", limit=prefetch_limit)
            )
            routes.append("bge_dense")
            
        # 路线 B：稀疏检索开关
        if use_text_routes and effective["use_sparse"] and "bge_sparse" in encoded:
            prefetch_queries.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=encoded["bge_sparse"]["indices"],
                        values=encoded["bge_sparse"]["values"]
                    ),
                    using="bge_sparse", limit=prefetch_limit
                )
            )
            routes.append("bge_sparse")

        # 路线 C：视觉检索开关
        if use_vision_route and vision_enabled and encoded.get("colpali_vision"):
            prefetch_queries.append(
                models.Prefetch(query=encoded["colpali_vision"], using="colpali_vision", limit=prefetch_limit)
            )
            routes.append("colpali_vision")

        if not prefetch_queries and retrieval_mode == "vision":
            print("  ⚠️ [模态降级] vision 未产生可用向量，降级为 text 检索。")
            retrieval_mode = "text"
            encoded = self.encoder.encode(
                query,
                text_only=True,
                retrieval_mode="text",
                use_hyde_override=effective["use_hyde"],
            )
            if effective["use_dense"] and "bge_dense" in encoded:
                prefetch_queries.append(
                    models.Prefetch(query=encoded["bge_dense"], using="bge_dense", limit=prefetch_limit)
                )
                routes.append("bge_dense")
            if effective["use_sparse"] and "bge_sparse" in encoded:
                prefetch_queries.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=encoded["bge_sparse"]["indices"],
                            values=encoded["bge_sparse"]["values"]
                        ),
                        using="bge_sparse", limit=prefetch_limit
                    )
                )
                routes.append("bge_sparse")

        self.last_search_metadata = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "text_only": text_only,
            "search_style": effective["requested_search_style"],
            "effective_search_style": effective["effective_search_style"],
            "effective_strategies": {
                "use_dense": effective["use_dense"],
                "use_sparse": effective["use_sparse"],
                "use_hyde": effective["use_hyde"],
                "use_vision": vision_enabled,
                "use_reranker": self.use_reranker,
            },
            "routes": routes,
            "fusion": "RRF",
            "notes": effective["notes"],
        }

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
        if self.use_reranker and points and retrieval_mode != "vision":
            sentence_pairs = []

            # 🚀 净化 1：对 Query 进行终极清洗
            clean_query = str(query).encode('utf-8', 'ignore').decode('utf-8')

            for point in points:
                # 从 payload 中拿回建库时存入的 markdown_text
                doc_text = point.payload.get("markdown_text", "")
                # 🛡️ 兜底：如果数据库里查出来的是 None，强转为空字符串
                if doc_text is None:
                    doc_text = ""

                # 🚀 净化 2：对数据库捞出来的文档也进行清洗
                clean_doc = str(doc_text).encode('utf-8', 'ignore').decode('utf-8')

                sentence_pairs.append([clean_query, clean_doc])
            
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
