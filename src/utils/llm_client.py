from openai import OpenAI
import os
import re
import base64
from dotenv import load_dotenv
load_dotenv()          # 读取 .env 文件，将内容注入 os.environ

class LLMClient:
    def __init__(self, api_config: dict):
        base_url = api_config.get("base_url")
        api_key = os.getenv("LLM_API_KEY")
        
        # 支持区分不同的模型用途
        self.reasoning_model = api_config.get("reasoning_model", "gpt-4o-mini")
        self.generation_model = api_config.get("generation_model", "gpt-4o")

        if not api_key:
            raise ValueError("❌ LLM API Key 未配置！请在 .env 中设置 LLM_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _encode_image(self, image_path: str) -> str:
        """将本地图片转换为 Base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, system_prompt: str, user_prompt: str, image_paths: list = None, temperature: float = 0.3, is_final_answer: bool = False, stream: bool = False):
        model_to_use = self.generation_model if is_final_answer else self.reasoning_model
        
        # ==========================================
        # 🛡️ 终极物理净化装甲：抹杀终端粘贴进来的所有幽灵字符
        # ==========================================
        # 使用 replace 替换掉无法识别的字符，绝不能让乱码流进字典！
        safe_sys = str(system_prompt).encode('utf-8', 'replace').decode('utf-8')
        safe_user = str(user_prompt).encode('utf-8', 'replace').decode('utf-8')

        # ==========================================
        # 适配 ：动态数据结构（防止纯文本模型崩溃）
        # ==========================================
        if image_paths and len(image_paths) > 0:
            # 只有在真有图片时，才使用多模态的 List/Dict 结构
            user_content = [{"type": "text", "text": safe_user}]
            for img_path in image_paths:
                if os.path.exists(img_path):
                    base64_img = self._encode_image(img_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
        else:
            # 如果没有图片，必须降维成纯字符串！否则 R1 等纯文本模型会直接报错
            user_content = safe_user

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": safe_sys},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            # 这里我帮你调回到了 800，避免报 Token 超限的错误
            max_tokens=4096 if is_final_answer else 1024,
            stream=stream # 将流式开关传给底层的 OpenAI SDK
        )
        
        # 如果开启了流式输出，返回一个 Python 生成器 (Generator)
        if stream:
            def chunk_generator():
                is_local_thinking = False # 兼容本地 vLLM 的正则开关
                
                for chunk in response:
                    choices = getattr(chunk, "choices", None) or []
                    if len(choices) > 0 and choices[0] is not None:
                        delta = getattr(choices[0], "delta", None)
                        if delta is None:
                            continue
                        
                        # 🌟 核心适配 4：优雅无视 SiliconCloud 的思考字段！
                        # 官方把思考过程放在了 delta.reasoning_content 里，我们根本不去读取它。
                        # 我们只专注读取最终答案字段 delta.content。
                        if hasattr(delta, 'content') and delta.content is not None:
                            text = delta.content
                            
                            # 🛡️ 兼容本地模型的擦除逻辑（以防你哪天又切回本地 vLLM）
                            if "<think>" in text:
                                is_local_thinking = True
                                text = text.replace("<think>", "")
                            if "</think>" in text:
                                is_local_thinking = False
                                text = text.replace("</think>", "")
                                
                            if not is_local_thinking and text.strip():
                                yield text
            return chunk_generator()
            
        # 否则保持原样，一次性返回全部字符串
        else:
            choices = getattr(response, "choices", None) or []
            if not choices or choices[0] is None:
                return ""

            message = getattr(choices[0], "message", None)
            raw_content = getattr(message, "content", None) if message is not None else None
            if raw_content is None:
                return ""

            raw_content = str(raw_content).strip()
            
            # 🛡️ 终极擦除：用正则精准抹除 <think> 到 </think> 之间的所有内容（包括换行）
            # re.DOTALL 允许 '.' 匹配换行符
            clean_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            # 如果模型偶尔没闭合标签，兜底处理一下
            if "<think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
                
            return clean_content
