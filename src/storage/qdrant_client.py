import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.utils.logger import logger

class QdrantManager:
    def __init__(self, host: str = "localhost", port: int = 6333, timeout: int = None):
        print("🔌 正在连接 Qdrant 向量数据库...")
        self.timeout = int(os.environ.get("QDRANT_TIMEOUT", timeout or 120))
        if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
            no_proxy_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
            current_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
            merged_no_proxy = current_no_proxy.split(",") if current_no_proxy else []
            for item in no_proxy_hosts:
                if item not in merged_no_proxy:
                    merged_no_proxy.append(item)
            os.environ["NO_PROXY"] = ",".join(filter(None, merged_no_proxy))
            os.environ["no_proxy"] = os.environ["NO_PROXY"]
        self.client = QdrantClient(host=host, port=port, timeout=self.timeout)
        print("✅ 数据库连接成功！")

    def create_collection_if_not_exists(self, collection_name: str = "omni_rag_docs"):
        """
        初始化集合，声明三路命名向量 (Named Vectors)。
        """
        if self.client.collection_exists(collection_name):
            print(f"ℹ️ 集合 '{collection_name}' 已存在，跳过创建。")
            return

        print(f"🔨 正在创建多模态集合 '{collection_name}'...")
        self.client.create_collection(
            collection_name=collection_name,
            # 1. 稠密向量 (Dense) - BGE-M3 (1024维)
            vectors_config={
                "bge_dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                ),
                # 2. 多向量 (Multi-Vector) - ColPali 视觉向量
                # ⚠️ 架构师注意：ColPali 每页会生成多个 Patch 向量 (N, 128)，必须开启 multivector_config
                "colpali_vision": models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                )
            },
            # 3. 稀疏向量 (Sparse) - BGE-M3 词频权重
            sparse_vectors_config={
                "bge_sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        print("✅ 集合创建完毕，三路向量通道已开启！")

    def upsert_points_batch(self, collection_name: str, points: List[models.PointStruct], batch_size: int = 10):
        """
        工业级批量写入。
        """
        print(f"🗄️ 准备将 {len(points)} 个节点打入 Qdrant...")
        
        # batch_size 不能过大，否则会报错（Qdrant限制每次网络传输文件最大为 32 MB）
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._upsert_with_split(collection_name, batch, i, len(points))
            print(f"✅ 成功入库: {i + len(batch)} / {len(points)}")

    def _upsert_with_split(
        self,
        collection_name: str,
        points: List[models.PointStruct],
        start_index: int,
        total_points: int,
    ):
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return
        except Exception as e:
            if len(points) > 1:
                mid = len(points) // 2
                logger.warning(
                    "⚠️ Qdrant 批量写入失败，自动拆分重试: "
                    f"批次 {start_index + 1}-{start_index + len(points)} / {total_points}, "
                    f"拆为 {mid} + {len(points) - mid}。原因: {e}"
                )
                self._upsert_with_split(collection_name, points[:mid], start_index, total_points)
                self._upsert_with_split(collection_name, points[mid:], start_index + mid, total_points)
                return

            logger.error(
                f"❌ Qdrant 写入失败 (节点 {start_index + 1}/{total_points}): {e}",
                exc_info=True
            )
            raise RuntimeError(
                "数据库写入崩溃: "
                f"{e}. 已将批次拆到单节点仍失败，请检查 Qdrant 服务、代理/NO_PROXY、"
                "磁盘/内存资源，或调小 configs/config.yaml 中 offline.database.batch_size。"
            )
