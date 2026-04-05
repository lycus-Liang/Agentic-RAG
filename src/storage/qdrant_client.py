from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        print("🔌 正在连接 Qdrant 向量数据库...")
        self.client = QdrantClient(host=host, port=port)
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
            
            # 🌟 加上 try-except 护甲
            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                print(f"✅ 成功入库: {i + len(batch)} / {len(points)}")
                
            except Exception as e:
                # 记录详细错误堆栈
                logger.error(f"❌ Qdrant 写入失败 (批次 {i} 到 {i+len(batch)}): {e}", exc_info=True)
                # 如果连数据库都写不进去，这个文件的后续处理没有意义了。
                # 必须把错误 raise 抛出去，让 main.py 外层的 except 捕获，从而安全跳过这本 PDF！
                raise RuntimeError(f"数据库写入崩溃: {e}")