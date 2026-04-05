import os
import pandas as pd
import random
import shutil
from pathlib import Path

def create_hardcore_sandbox(
    csv_path="officeqa/officeqa_pro.csv", 
    pdf_source_dir="officeqa/treasury_bulletin_pdfs", 
    target_dir="./data/raw/officeqa_test",
    num_questions=20,
    noise_multiplier=5  # 每 1 份目标 PDF，混入 5 份干扰 PDF (可根据 AutoDL 算力调大)
):
    print("🏗️ 正在构建高密度噪音沙盒...")
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. 读取官方题库
    df = pd.read_csv(csv_path)
    
    # 2. 随机抽取 N 道题作为目标测试集
    sampled_df = df.sample(n=num_questions, random_state=42)
    sampled_df.to_csv("officeqa_test_targets.csv", index=False)
    print(f"✅ 成功抽取 {num_questions} 道测试题，已保存至 officeqa_test_targets.csv")
    
    # 3. 提取这 N 道题对应的目标 PDF
    target_pdfs = set()
    for files_str in sampled_df['source_files']:
        # OfficeQA 的 source_files 可能是列表字符串，如 "['file1.pdf', 'file2.pdf']"
        # 简单清理提取
        clean_str = files_str.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
        files = [f.strip() for f in clean_str.split(",")]
        target_pdfs.update(files)
        
    print(f"🎯 识别到 {len(target_pdfs)} 份目标 PDF。")
    
    # 4. 获取所有可用的 PDF，剥离出“纯噪音池”
    all_pdfs = set(f for f in os.listdir(pdf_source_dir) if f.endswith(".pdf"))
    noise_pool = list(all_pdfs - target_pdfs)
    
    # 5. 抽取海量噪音 PDF
    num_noise = min(len(target_pdfs) * noise_multiplier, len(noise_pool))
    sampled_noise = random.sample(noise_pool, num_noise)
    print(f"😈 抽取到 {num_noise} 份纯干扰 PDF (噪音比例 1:{noise_multiplier})。")
    
    # 6. 将目标和噪音全部拷贝到测试目录
    final_pdfs_to_copy = list(target_pdfs) + sampled_noise
    
    for pdf_name in final_pdfs_to_copy:
        src = os.path.join(pdf_source_dir, pdf_name)
        dst = os.path.join(target_dir, pdf_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            
    print(f"🎉 沙盒构建完成！共计 {len(final_pdfs_to_copy)} 份硬核财报 PDF 已就绪：{target_dir}")

if __name__ == "__main__":
    # 请确保你已经把 officeqa 仓库克隆到了当前目录下
    create_hardcore_sandbox(noise_multiplier=10) # 👈 1:10 比例，大概两三百份PDF，几万页！