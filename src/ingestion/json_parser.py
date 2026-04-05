import json
import os
from typing import List
from src.utils.logger import logger

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
        extracted_texts.append(combined_text)

    logger.info(f"✅ 成功提取并缝合了 {len(extracted_texts)} 道完整的 QA 语料！")
    return extracted_texts