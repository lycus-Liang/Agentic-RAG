import fitz
import os
import multiprocessing
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

def worker_process_batch(pdf_path: str, output_base_dir: str, page_batch: List[int], dpi: int) -> List[Dict[str, str]]:
    """
    子进程的工作函数：专门负责处理分配给它的一小批页码。
    """
    pdf_file = Path(pdf_path)
    doc_name = pdf_file.stem
    pdf_out_dir = Path(output_base_dir) / doc_name / "pdfs"
    img_out_dir = Path(output_base_dir) / doc_name / "images"
    
    # 每个子进程独立打开文件 (PyMuPDF 底层用的是内存映射 mmap，极其高效，不用担心爆内存)
    doc = fitz.open(pdf_path)
    batch_records = []

    for page_num in page_batch:
        current_page_idx = page_num + 1
        
        single_page_pdf_path = pdf_out_dir / f"page_{current_page_idx}.pdf"
        single_page_img_path = img_out_dir / f"page_{current_page_idx}.png"

        # 1. 提取单页 PDF
        single_doc = fitz.open()
        single_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        single_doc.save(str(single_page_pdf_path))
        single_doc.close()

        # 2. 渲染高清 PNG
        page = doc.load_page(page_num)
        zoom_mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=zoom_mat, alpha=False)
        pix.save(str(single_page_img_path))

        batch_records.append({
            "page_number": current_page_idx,
            "pdf_path": str(single_page_pdf_path),
            "image_path": str(single_page_img_path)
        })

    doc.close()
    return batch_records

def split_and_render_pdf_parallel(pdf_path: str, output_base_dir: str, dpi: int = 300, batch_size: int = 50, workers = None) -> List[Dict[str, str]]:
    """
    主控函数：使用多进程榨干 CPU 核心，极速处理 GB 级 PDF。
    注意：workers 的数量最好小于 8，否则会导致子进程被 kill
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"找不到指定的 PDF 文件: {pdf_path}")

    doc_name = pdf_file.stem
    pdf_out_dir = Path(output_base_dir) / doc_name / "pdfs"
    img_out_dir = Path(output_base_dir) / doc_name / "images"
    pdf_out_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # 仅在主进程快速获取总页数
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()
    print(f"🚀 开始并发处理文档: {pdf_file.name} (共 {total_pages} 页)")

    # 把数十万页拆分成多个 Batch，比如每 50 页一个包
    page_batches = [list(range(i, min(i + batch_size, total_pages))) for i in range(0, total_pages, batch_size)]
    
    all_records = []

    # 如果未指定进程数，则自动获取机器的最佳进程数 (留一个核心给系统，防止电脑卡死)
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
        
    print(f"⚡ 启动进程池，使用 {workers} 个 Worker，分 {len(page_batches)} 批处理...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(worker_process_batch, pdf_path, output_base_dir, batch, dpi): batch 
            for batch in page_batches
        }

        # 收集结果 (带进度展示)
        completed_batches = 0
        for future in as_completed(futures):
            try:
                batch_result = future.result()
                all_records.extend(batch_result)
                completed_batches += 1
                print(f"  -> 进度: {completed_batches}/{len(page_batches)} 批次完成 (已处理 {len(all_records)} 页)")
            except Exception as e:
                print(f"❌ 处理某个批次时发生错误: {e}")

    # 按照页码对结果进行重新排序，因为并发返回的顺序是乱的
    all_records.sort(key=lambda x: x["page_number"])
    
    print(f"✅ 全部分布式处理完成！成功渲染 {len(all_records)} 页。")
    return all_records

if __name__ == "__main__":
    # 使用测试
    # SAMPLE_PDF = "./data/raw/普通高中教科书语文必修.pdf" 
    SAMPLE_PDF = "./data/raw/RAG调研.pdf"
    OUTPUT_DIR = "./data/processed"
    
    if os.path.exists(SAMPLE_PDF):
        # 建议 batch_size 设为 50-100，太大容易在进程间传递结果时卡顿
        records = split_and_render_pdf_parallel(SAMPLE_PDF, OUTPUT_DIR, dpi=300, batch_size=50)
    else:
        print("准备好 PDF 文件后即可测试并发性能！")