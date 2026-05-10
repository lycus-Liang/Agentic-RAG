import os
import re
import base64
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
load_dotenv()          # 读取 .env 文件，将内容注入 os.environ

class LLMClient:
    def __init__(self, api_config: dict):
        base_url = api_config.get("base_url")
        api_key = os.getenv("LLM_API_KEY")
        self.base_url = base_url
        
        # 支持区分不同的模型用途
        self.reasoning_model = api_config.get("reasoning_model", "gpt-4o-mini")
        self.generation_model = api_config.get("generation_model", "gpt-4o")

        if not api_key:
            raise ValueError("❌ LLM API Key 未配置！请在 .env 中设置 LLM_API_KEY")

        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _safe_text(self, value) -> str:
        return str(value).encode('utf-8', 'replace').decode('utf-8')

    def _message_to_dict(self, message) -> dict:
        if hasattr(message, "model_dump"):
            data = message.model_dump(exclude_none=True)
        elif isinstance(message, dict):
            data = dict(message)
        else:
            data = {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", None),
                "tool_calls": getattr(message, "tool_calls", None),
            }

        tool_calls = data.get("tool_calls") or []
        normalized_tool_calls = []
        for call in tool_calls:
            if hasattr(call, "model_dump"):
                call_data = call.model_dump(exclude_none=True)
            else:
                call_data = dict(call)
            function_data = call_data.get("function") or {}
            if hasattr(function_data, "model_dump"):
                function_data = function_data.model_dump(exclude_none=True)
            normalized_tool_calls.append({
                "id": call_data.get("id"),
                "type": call_data.get("type", "function"),
                "function": {
                    "name": function_data.get("name"),
                    "arguments": function_data.get("arguments", "{}"),
                },
            })

        return {
            "role": data.get("role", "assistant"),
            "content": data.get("content") or "",
            "tool_calls": normalized_tool_calls,
        }

    def _sanitize_tool_messages(self, messages: list) -> list:
        sanitized = []
        for message in messages or []:
            role = message.get("role")
            if role == "assistant":
                assistant_message = {
                    "role": "assistant",
                    "content": self._safe_text(message.get("content", "")),
                }
                if message.get("tool_calls"):
                    assistant_message["tool_calls"] = message["tool_calls"]
                sanitized.append(assistant_message)
            elif role == "tool":
                sanitized.append({
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": self._safe_text(message.get("content", "")),
                })
            else:
                sanitized.append({
                    "role": role or "user",
                    "content": self._safe_text(message.get("content", "")),
                })
        return sanitized

    def chat_with_tools(
        self,
        system_prompt: str,
        messages: list,
        tools: list,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> dict:
        """Call an OpenAI-compatible chat model with native tool calling enabled."""
        if not tools:
            raise ValueError("tool calling requires at least one tool schema")

        api_messages = [
            {"role": "system", "content": self._safe_text(system_prompt)}
        ] + self._sanitize_tool_messages(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=api_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(
                "LLM tool calling request failed. "
                f"model={self.reasoning_model}, base_url={self.base_url}, error={e}"
            ) from e

        choices = getattr(response, "choices", None) or []
        if not choices or choices[0] is None:
            raise RuntimeError(
                "LLM tool calling response did not contain choices. "
                f"model={self.reasoning_model}, base_url={self.base_url}"
            )

        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError(
                "LLM tool calling response did not contain an assistant message. "
                f"model={self.reasoning_model}, base_url={self.base_url}"
            )

        return self._message_to_dict(message)

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
        safe_sys = self._safe_text(system_prompt)
        safe_user = self._safe_text(user_prompt)

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
