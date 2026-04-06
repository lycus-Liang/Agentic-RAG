import json
import os
from typing import List
from src.utils.logger import logger

def deep_clean_text(raw_text: str) -> str:
    """
    🧹 深度清洗文本：去除所有可能导致底层 C++/Rust 引擎及 JSON 序列化崩溃的幽灵字符和非法代理对。
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    
    # 将文本转化为纯正的 UTF-8，忽略掉所有非法/幽灵字符，然后再解密回字符串
    return raw_text.encode('utf-8', 'ignore').decode('utf-8').strip()

def _safe_extract_text(value) -> str:
    """
    🛡️ 万能安全提取器：无论传入的是什么鬼格式，都把它榨成干净的字符串
    """
    if not value:
        return ""
    
    # 正常情况：如果是字符串，直接 strip
    if isinstance(value, str):
        return value.strip()
    
    # 异常情况 1：如果是列表（比如把多段落存成了 List），用换行符拼起来
    if isinstance(value, list):
        # 递归处理列表里的每一个元素，并过滤掉空值
        cleaned_list = [str(v).strip() for v in value if v]
        return "\n".join(cleaned_list)
    
    # 异常情况 2：如果是数字或字典，强转为字符串
    return str(value).strip()


def parse_json_corpus(json_path: str) -> List[str]:
    """
    专门针对结构化问答 JSON 的解析器 (带终极防御版)
    """
    logger.info(f"📄 开始解析 JSON 结构化语料: {os.path.basename(json_path)}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"❌ JSON 加载失败: {e}")
        raise

    # 1. 兼容性探测：定位核心列表
    records = []
    if isinstance(data, dict) and "example" in data:
        records = data["example"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON 结构不符：期望是一个列表，或者包含 'example' 键的字典。")

    # 2. 字段重组缝合
    extracted_texts = []
    for item in records:
        if not isinstance(item, dict):
            continue
            
        # 🌟 核心防御：调用刚刚写好的 _safe_extract_text 函数！
        q = _safe_extract_text(item.get("question"))
        a = _safe_extract_text(item.get("answer"))
        analysis = _safe_extract_text(item.get("analysis"))
        
        if not q:
            continue
            
        combined_text = f"【题目】\n{q}\n\n【答案】\n{a}\n\n【解析】\n{analysis}"
        # 源头净化：在装入列表前，深度清洗掉所有的幽灵字符
        clean_text = deep_clean_text(combined_text)
        extracted_texts.append(clean_text)

    logger.info(f"✅ 成功提取并缝合了 {len(extracted_texts)} 道完整的 QA 语料！")
    return extracted_texts


def parse_alpaca_jsonl(jsonl_path: str) -> List[str]:
    """
    专门解析带有 instruction, input, output 格式的 JSONL 文件
    """
    logger.info(f"📄 开始解析 JSONL 微调语料: {os.path.basename(jsonl_path)}")
    
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"文件不存在: {jsonl_path}")

    extracted_texts = []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    # 提取三大核心字段
                    instruction = data.get("instruction", "").strip()
                    input_text = data.get("input", "").strip()
                    output_text = data.get("output", "").strip()
                    
                    # 过滤掉完全没有有效信息的行
                    if not instruction and not input_text:
                        continue
                        
                    # 🌟 缝合成标准 Markdown 文本块
                    combined_text = f"【指令】\n{instruction}\n"
                    if input_text:
                        combined_text += f"\n【输入】\n{input_text}\n"
                    if output_text:
                        combined_text += f"\n【输出】\n{output_text}\n"

                    # 源头净化：在装入列表前，深度清洗掉所有的幽灵字符
                    clean_text = deep_clean_text(combined_text.strip())
                    extracted_texts.append(clean_text)
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ 第 {line_num} 行 JSON 格式损坏，已跳过。")
                    
    except Exception as e:
        logger.error(f"❌ JSONL 读取失败: {e}")
        raise

    logger.info(f"✅ 成功提取并缝合了 {len(extracted_texts)} 条 JSONL 记录！")
    return extracted_texts