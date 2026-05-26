from __future__ import annotations

import json
from typing import Any

import httpx


class OpenAICompatibleError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            self._chat_url = normalized
        elif normalized.endswith("/v1"):
            self._chat_url = f"{normalized}/chat/completions"
        else:
            self._chat_url = f"{normalized}/v1/chat/completions"

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key,
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(self._chat_url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise OpenAICompatibleError(
                f"模型接口请求失败: HTTP {response.status_code} {response.text[:500]}"
            )
        data = response.json()
        if not data.get("choices"):
            raise OpenAICompatibleError(
                f"模型接口响应缺少 choices: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return data

    @staticmethod
    def extract_text_content(response_json: dict[str, Any]) -> str:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    collected.append(str(item.get("text", "")))
            return "\n".join(part for part in collected if part)
        return str(content)


class AsyncOpenAICompatibleClient:
    """异步 OpenAI 兼容客户端。"""
    
    def __init__(self, *, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            self._chat_url = normalized
        elif normalized.endswith("/v1"):
            self._chat_url = f"{normalized}/chat/completions"
        else:
            self._chat_url = f"{normalized}/v1/chat/completions"

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key,
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(self._chat_url, headers=headers, json=payload)
        if response.status_code >= 400:
            raise OpenAICompatibleError(
                f"模型接口请求失败: HTTP {response.status_code} {response.text[:500]}"
            )
        data = response.json()
        if not data.get("choices"):
            raise OpenAICompatibleError(
                f"模型接口响应缺少 choices: {json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return data

    @staticmethod
    def extract_text_content(response_json: dict[str, Any]) -> str:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    collected.append(str(item.get("text", "")))
            return "\n".join(part for part in collected if part)
        return str(content)
