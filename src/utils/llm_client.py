from openai import OpenAI
import os
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
            raise ValueError("❌ LLM API Key 未配置！")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _encode_image(self, image_path: str) -> str:
        """将本地图片转换为 Base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate(self, system_prompt: str, user_prompt: str, image_paths: list = None, temperature: float = 0.3, is_final_answer: bool = False, stream: bool = False):
        model_to_use = self.generation_model if is_final_answer else self.reasoning_model
        
        user_content = [{"type": "text", "text": user_prompt}]
        
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    base64_img = self._encode_image(img_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=1024 if is_final_answer else 256,
            stream=stream # 🌟 将流式开关传给底层的 OpenAI SDK
        )
        
        # 如果开启了流式输出，返回一个 Python 生成器 (Generator)
        if stream:
            def chunk_generator():
                for chunk in response:
                    # 🛡️ 防御性编程：检查 choices 是否为空
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        # 🛡️ 检查 delta 中是否有 content 且不为 None
                        if hasattr(delta, 'content') and delta.content is not None:
                            yield delta.content
            return chunk_generator()
            
        # 否则保持原样，一次性返回全部字符串
        else:
            return response.choices[0].message.content.strip()