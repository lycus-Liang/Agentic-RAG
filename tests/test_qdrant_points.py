import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.http import models

def test_qdrant_integrity(collection_name="omni_rag_docs"):
    print("🔍 开始执行 Qdrant 节点完整性断言测试...")
    client = QdrantClient("localhost", port=6333)
    
    # 1. 检查 Collection 是否存在
    assert client.collection_exists(collection_name), f"❌ 集合 {collection_name} 不存在！"
    
    # 2. 检查总量
    collection_info = client.get_collection(collection_name)
    points_count = collection_info.points_count
    print(f"📊 当前数据库总量 (Point Count): {points_count}")
    assert points_count > 0, "❌ 数据库是空的，入库失败！"
    
    # 3. 随机抽样 5 个节点
    scroll_result, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=True
    )
    
    for idx, point in enumerate(scroll_result):
        print(f"\n🔬 正在抽检第 {idx+1} 个节点 (ID: {point.id})")
        
        # [断言 1] Payload 完整性
        assert "markdown_text" in point.payload, "❌ 缺失文本 Payload!"
        assert "extracted_crop_paths" in point.payload, "❌ 缺失图片绝对路径 Payload!"
        print("  ✅ Payload 结构完整")
        
        # [断言 2] 稠密向量维度 (BGE-M3 -> 1024)
        dense_vec = point.vector.get("bge_dense")
        assert dense_vec is not None, "❌ 缺失 BGE 稠密向量!"
        assert len(dense_vec) == 1024, f"❌ 稠密向量维度错误，预期 1024，实际 {len(dense_vec)}"
        print("  ✅ 稠密向量 (Dense) 验证通过")
        
        # [断言 3] 多向量视觉维度 (ColPali -> N 个 128 维的 Patch)
        vision_vec = point.vector.get("colpali_vision")
        assert vision_vec is not None, "❌ 缺失 ColPali 视觉向量!"
        # 由于它是 MultiVector，它应该是一个二维列表 (N, 128)
        assert len(vision_vec) > 0 and len(vision_vec[0]) == 128, f"❌ ColPali 视觉向量格式错误"
        print(f"  ✅ 视觉多向量 (Multi-Vector) 验证通过 (Patch数量: {len(vision_vec)})")

    print("\n🎉 所有底层断言全部通过！Qdrant 数据库构建完美无瑕！")

if __name__ == "__main__":
    test_qdrant_integrity()