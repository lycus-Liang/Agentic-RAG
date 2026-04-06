import os
import argparse
import yaml
import glob
import concurrent.futures
from pathlib import Path

# 🌟 架构师铁律：在导入任何底层大模型或 RAG 库之前，先加载环境变量！
from dotenv import load_dotenv
load_dotenv(override=True)
print(f"🌍 当前 HF 镜像源: {os.environ.get('HF_ENDPOINT', '官方默认')}")
# ==========================================

# [离线链路] Ingestion 组件
# ==========================================
from src.ingestion.splitter import split_and_render_pdf_parallel
from src.ingestion.vision_worker import VisionWorker
from src.ingestion.text_worker import TextWorker
from src.ingestion.proxy_worker import ProxyWorker
from src.ingestion.json_parser import parse_json_corpus, parse_alpaca_jsonl
from src.storage.qdrant_client import QdrantManager
from src.storage.point_builder import PointBuilder

from src.utils.logger import logger
from src.utils.cache import run_with_cache

PARSER_REGISTRY = {
    "json": parse_json_corpus,
    "jsonl": parse_alpaca_jsonl,
    # 未来可以扩展：
    # "csv": parse_csv_to_text,
    # "txt": parse_plain_text,
}

def get_config() -> dict:
    """加载全局配置"""
    config_path = "./configs/config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_text_pipeline(file_path: str, file_type: str, workers: dict, db_manager: QdrantManager, config: dict):
    """
    纯文本 JSON 专属建库流水线
    """
    file_name = os.path.basename(file_path)
    print(f"\n🚀 [阶段一] 根据 {file_type} 策略读取并解析文档: {file_name}...")
    
    # 🌟 动态调用对应的解析器
    parser_func = PARSER_REGISTRY.get(file_type)
    if not parser_func:
        print(f"❌ 不支持的文件类型: {file_type}，请在 PARSER_REGISTRY 中注册！")
        return

    try:
        markdown_texts = parser_func(file_path)
    except Exception as e:
        print(f"❌ 解析失败，跳过建库。原因: {e}")
        return


    text_worker = workers["text"]
    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
    batch_size = config.get("offline", {}).get("database", {}).get("batch_size", 16) # 纯文本 batch

    print(f"\n🚀 [阶段二] 极速提取文本双路向量 (BGE-M3)...")
    final_points = []
    
    for i in range(0, len(markdown_texts), batch_size):
        batch_texts = markdown_texts[i:i+batch_size]
        
        # 批量提向量
        embeddings = text_worker.embed_model.encode(
            batch_texts, return_dense=True, return_sparse=True, batch_size=batch_size
        )
        
        print(f"  -> 向量化进度: {min(i + len(batch_texts), len(markdown_texts))}/{len(markdown_texts)}")
        
        # 组装节点 (彻底物理隔离视觉特征)
        for j in range(len(batch_texts)):
            point = PointBuilder.build_point(
                source_doc=file_name,
                page_num=i + j + 1, # 这里 page_num 代表第几道题
                markdown_text=batch_texts[j],
                dense_vec=embeddings['dense_vecs'][j].tolist(),
                sparse_vec={str(k): float(v) for k, v in embeddings['lexical_weights'][j].items()},
                vision_multivec=None, # 🚫 彻底置空
                crop_paths=[],        # 🚫 彻底置空
                proxy_descriptions=[] # 🚫 彻底置空
            )
            final_points.append(point)
            
    print("\n🚀 [阶段三] 准备将纯文本数据轰入 Qdrant...")
    db_manager.upsert_points_batch(collection_name, final_points, batch_size=batch_size)
    print(f"🎉 《{file_name}》 纯文本极速建库完毕！")

def build_text_directory(dir_path: str, file_type: str, clear_db: bool = False):
    """
    批量处理文件夹下的所有 JSON 文件
    """
    json_files = glob.glob(os.path.join(dir_path, "*.json"))
    if not json_files:
        print(f"❌ 在目录 {dir_path} 下没有找到任何 JSON 文件！")
        return
        
    print(f"\n📦 发现 {len(json_files)} 个 JSON 文件。准备进行纯文本批量建库...")
    
    # 因为是纯文本，完全不需要初始化 VisionWorker 和 ProxyWorker！极大地节省了显存！
    workers = {
        "text": TextWorker(),
    }
    db_manager = QdrantManager()
    config = get_config()
    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
    
    if clear_db:
        print(f"💣 警告：正在清空原有的集合 [{collection_name}] ...")
        db_manager.client.delete_collection(collection_name)
        print("✅ 清理完毕！")
        
    db_manager.create_collection_if_not_exists(collection_name)
    
    for idx, json_path in enumerate(json_files, 1):
        print(f"\n🔄 处理 {idx}/{len(json_files)}: {os.path.basename(json_path)}")
        build_text_pipeline(json_path, file_type, workers, db_manager, config)
        
    print("\n🎉 JSON 批量建库作战全部结束！")

def build_pdf_pipeline(pdf_path: str, workers: dict, db_manager: QdrantManager, config: dict, batch_name: str = None):
    """端到端建库流水线 (适配多进程 Splitter)
       一次仅读取一个pdf
    """
    # 0. 准备工作目录与基础信息
    base_work_dir = "./data/processed"

    if batch_name:
        work_dir = os.path.join(base_work_dir, batch_name)
    else:
        work_dir = base_work_dir

    os.makedirs(work_dir, exist_ok=True)
    doc_name = Path(pdf_path).stem

    doc_work_dir = os.path.join(work_dir, doc_name)
    os.makedirs(doc_work_dir, exist_ok=True)
    
    # 1. 提取模型矩阵与数据库
    vision_worker = workers["vision"]
    text_worker = workers["text"]
    proxy_worker = workers["proxy"]
    
    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
    batch_size = config.get("offline", {}).get("database", {}).get("batch_size", 10)
    
    # 2. 物理拆解文档
    print("\n🚀 [阶段一] 呼叫 Splitter 进行文档物理多进程拆解...")
    step1_cache = os.path.join(doc_work_dir, "step1_records.pkl")

    records = run_with_cache(
        step1_cache, 
        split_and_render_pdf_parallel, 
        pdf_path, work_dir, dpi=150, batch_size=10, workers=4
    )
    
    single_pdf_paths = [rec["pdf_path"] for rec in records]
    single_image_paths = [rec["image_path"] for rec in records]
    
    # 3. [终极并发] 开启双主线并行：图文线 vs 全局视觉线
    print("\n🚀 [阶段二] 正在双线并发提取底层特征 (Text & Vision)...")
    step2_cache = os.path.join(doc_work_dir, "step2_text_payloads.pkl")
    step3_cache = os.path.join(doc_work_dir, "step3_vision_embeddings.pkl")
    # 使用 2 个线程，分别跑两条不同的业务大线
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        
        # 提交任务一：TextWorker 去跑 Omniparse 和 BGE-M3
        future_text = executor.submit(
            run_with_cache, 
            step2_cache,
            text_worker.process_pdf_batch,
            single_pdf_paths, doc_work_dir, batch_size=16
        )
        
        # 提交任务二：VisionWorker 去跑 ColPali
        future_vision = executor.submit(
            run_with_cache, 
            step3_cache,
            vision_worker.process_image_batch,
            single_image_paths, batch_size=4
        )
        
        # 阻塞等待两位大将全部凯旋！
        # 只要任何一个报错，这里会直接抛出异常，触发外层保护机制
        text_payloads = future_text.result()
        vision_embeddings = future_vision.result()

    print("✅ 双线特征提取完毕！")
    
    # 4. 局部图像代理翻译 & 数据组装
    print("\n🚀 [阶段三] VLM 翻译与终极节点组装...")
    step4_cache = os.path.join(doc_work_dir, "step4_final_points.pkl")

    def build_points_logic():
        points = []
        for idx, payload in enumerate(text_payloads):
            page_num = records[idx]["page_number"]
            extracted_crops = payload.get("extracted_crops", [])
            proxy_descriptions = []
            
            if extracted_crops:
                proxy_descriptions = proxy_worker.generate_proxy_batch(
                    extracted_crops, 
                    batch_size=4, 
                    page_num=page_num 
                )
                
            point = PointBuilder.build_point(
                source_doc=doc_name, 
                page_num=page_num, 
                markdown_text=payload["markdown"],
                dense_vec=payload["dense_vec"], 
                sparse_vec=payload["sparse_vec"],
                vision_multivec=vision_embeddings[idx], 
                crop_paths=extracted_crops,
                proxy_descriptions=proxy_descriptions
            )
            points.append(point)
        return points

    final_points = run_with_cache(step4_cache, build_points_logic)
        
    # 5. 轰入数据库！
    print("\n🚀 [阶段四] 目标锁定，准备轰入 Qdrant 数据库...")
    # 传入配置好的安全 batch_size
    db_manager.upsert_points_batch(collection_name, final_points, batch_size=batch_size)
    
    print(f"\n🎉 恭喜！《{doc_name}》全链路建库完毕！共入库 {len(final_points)} 页数据！")

def build_pdf_directory(dir_path: str, clear_db: bool = False):
    """批量建库：自动遍历并处理整个文件夹下的所有 PDF"""
    # 查找目录下所有的 .pdf 文件（排序）
    pdf_files = sorted(glob.glob(os.path.join(dir_path, "*.pdf")))
    
    if not pdf_files:
        print(f"❌ 在目录 {dir_path} 下没有找到任何 PDF 文件！")
        return

    batch_name = os.path.basename(os.path.normpath(dir_path))
        
    print(f"\n📦 扫描完毕！共发现 {len(pdf_files)} 个 PDF 文件，正在全局预热 AI 模型矩阵 (仅加载一次)...")
    workers = {
        "vision": VisionWorker(),
        "text": TextWorker(),
        "proxy": ProxyWorker()
    }
    db_manager = QdrantManager()
    config = get_config()
    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
    # 清理逻辑
    if clear_db:
        print(f"💣 警告：正在清空原有的集合 [{collection_name}] ...")
        db_manager.client.delete_collection(collection_name)
        print("✅ 清理完毕！")
    db_manager.create_collection_if_not_exists(config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs"))
    
    # 挨个处理，并加上容错机制 (try-except)
    for idx, pdf_path in enumerate(pdf_files, 1):
        print("\n" + "🔥"*30)
        print(f"🔄 正在处理第 {idx}/{len(pdf_files)} 个文件: {os.path.basename(pdf_path)}")
        print("🔥"*30)
        try:
            build_pdf_pipeline(pdf_path, workers, db_manager, config, batch_name)
            
        except Exception as e:
            # 防御性编程：如果某一本 PDF 损坏了，只跳过它，绝不让整个流水线崩溃！
            logger.error(f"\n❌ [致命错误] 文件 {os.path.basename(pdf_path)} 处理失败，已跳过。原因: {e}", exc_info=True)
            continue
            
    print("\n🎉 批量建库结束！")

def chat_loop(text_only: bool = False):
    """端到端在线交互终端 (Agentic RAG)"""
    from src.agent.workflow import build_agentic_rag

    print("\n" + "="*60)
    print("🧠 正在唤醒 Omni-Modal Agentic RAG 核心...")
    print("="*60)
    
    # 编译 Agent 状态机
    app = build_agentic_rag()
    
    print("\n✅ 智能体已就绪！")
    print("💡 提示：输入 'exit' 或 'quit' 退出, 输入 'clear' 清屏。\n")
    
    while True:
        try:
            user_input = input("🧑‍💻 你: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("👋 再见！")
                break
            if user_input.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue

            inputs = {
                "question": user_input,
                "text_only": text_only
            }
            print("\n🤖 Agent 思考与检索中...")

            # 流式运行 Agent
            for output in app.stream(inputs):
                for key, value in output.items():
                    # 🌟 优化日志输出：如果是 generate 节点，加个特殊提示
                    if key == "generate":
                        print(f"  👉 节点 [{key}] 动作完成 (流式输出完毕)。")
                    else:
                        print(f"  👉 节点 [{key}] 动作完成。")

            print("\n" + "-"*60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生异常: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Omni-Modal RAG 控制台")
    # command 选项扩展：支持 build 和 chat
    parser.add_argument("command", choices=["build", "chat"], help="执行的操作: build (离线建库) 或 chat (在线对话)")

    # 针对 Build (建库) 的指令
    parser.add_argument("--file", type=str, required=False, help="要处理的单个 PDF 文件路径")
    parser.add_argument("--dir", type=str, required=False, help="要批量处理的 PDF 文件夹路径")
    parser.add_argument("--clear", action="store_true", help="建库前是否清空旧的 Qdrant 集合")
    parser.add_argument("--type", choices=["pdf", "json", "jsonl"], default="pdf", help="指定建库的数据源类型")

    # 针对 Chat (对话) 的指令
    parser.add_argument("--text-only", action="store_true", help="对话时强制关闭视觉检索，仅使用文本双路召回")

    args = parser.parse_args()
    
    if args.command == "build":
        if args.type == "pdf":
            # 核心分流逻辑：传了 dir 就走批量，传了 file 就走单文件
            if args.dir:
                if os.path.isdir(args.dir):
                    build_pdf_directory(args.dir, args.clear)
                else:
                    print(f"❌ 找不到文件夹: {args.dir}，请检查路径是否正确！")
                
            elif args.file:
                if os.path.exists(args.file):
                    workers = {
                        "vision": VisionWorker(),
                        "text": TextWorker(),
                        "proxy": ProxyWorker()
                    }
                    db_manager = QdrantManager()
                    config = get_config()
                    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
                    # 清理逻辑
                    if args.clear:
                        print(f"💣 警告：正在清空原有的集合 [{collection_name}] ...")
                        db_manager.client.delete_collection(collection_name)
                        print("✅ 清理完毕！")
                    db_manager.create_collection_if_not_exists(config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs"))
                    build_pdf_pipeline(args.file, workers, db_manager, config)
                else:
                    print(f"❌ 找不到文件: {args.file}，请检查路径是否正确！")
                    
            else:
                print("❌ 命令错误！请提供正确的参数：\n批量建库: python main.py build --dir <文件夹路径>\n单文件建库: python main.py build --file <文件路径>")

        elif args.type in PARSER_REGISTRY:
            if args.dir:
                if os.path.isdir(args.dir):
                    build_text_directory(args.dir, args.clear)
                else:
                    print(f"❌ 找不到文件夹: {args.dir}，请检查路径是否正确！")
                
            elif args.file:
                if os.path.exists(args.file):
                    workers = {
                        "text": TextWorker(),
                    }
                    db_manager = QdrantManager()
                    config = get_config()
                    collection_name = config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs")
                    # 清理逻辑
                    if args.clear:
                        print(f"💣 警告：正在清空原有的集合 [{collection_name}] ...")
                        db_manager.client.delete_collection(collection_name)
                        print("✅ 清理完毕！")
                    db_manager.create_collection_if_not_exists(config.get("offline", {}).get("database", {}).get("collection_name", "omni_rag_docs"))
                    build_text_pipeline(args.file, args.type, workers, db_manager, config)
                else:
                    print(f"❌ 找不到文件: {args.file}，请检查路径是否正确！")
                    
            else:
                print("❌ 命令错误！请提供正确的参数：\n批量建库: python main.py build --dir <文件夹路径>\n单文件建库: python main.py build --file <文件路径>")

    elif args.command == "chat":
        if args.text_only:
            print("⚠️ 注意：已开启 [纯文本极速模式]，视觉检索模块 (ColPali) 已被物理隔离！")
        chat_loop(text_only=args.text_only)