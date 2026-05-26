from __future__ import annotations

import base64

from google import genai
from google.genai import types


class AsyncGeminiMultimodalClient:
    """异步 Gemini 多模态客户端，用于图文融合分析。"""

    def __init__(self, *, api_key: str, model: str, timeout_seconds: float = 180.0) -> None:
        self._model = model
        self._api_key = api_key

    async def generate_content(
        self,
        *,
        prompt_text: str,
        image_data_uris: list[str],
        max_output_tokens: int = 1200,
    ) -> str:
        """使用 Gemini 对文本和图片进行多模态分析，返回分析结果文本。"""
        parts: list[types.Part] = []

        for data_uri in image_data_uris:
            if data_uri.startswith("data:"):
                header, encoded = data_uri.split(",", 1)
                mime_type = header.split(";")[0][5:]  # 去掉 "data:" 前缀
                image_bytes = base64.b64decode(encoded)
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        parts.append(types.Part(text=prompt_text))

        # 每次调用时创建新的 client，使用 async with 确保异步 HTTP 连接正确释放
        client = genai.Client(api_key=self._api_key)
        async with client.aio:
            response = await client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                ),
            )
            return response.text or ""