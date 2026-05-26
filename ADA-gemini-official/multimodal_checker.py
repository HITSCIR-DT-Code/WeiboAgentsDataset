from __future__ import annotations

import asyncio
import base64
import mimetypes
import sys
from datetime import datetime
from typing import Any

import httpx

from config import AppConfig
from GeminiAPI import AsyncGeminiMultimodalClient
from models import ImageAnalysisResult, WeiboPost


class MultimodalConsistencyChecker:
    def __init__(self, config: AppConfig, client: AsyncGeminiMultimodalClient) -> None:
        self._config = config
        self._client = client

    async def analyze_posts(self, posts: list[WeiboPost], image_dir: str = "") -> list[ImageAnalysisResult]:
        # 筛选出有图片的帖子
        posts_with_images = [post for post in posts if post.images]
        
        if not posts_with_images:
            return []
        
        # 在当前事件循环中创建 Semaphore，避免跨 event loop 绑定问题
        semaphore = asyncio.Semaphore(3)
        total = len(posts_with_images)
        print(f"  开始异步分析 {total} 个帖子（并发数: 3）...", flush=True)
        
        # 创建所有异步任务
        tasks = [
            self._analyze_post_with_semaphore(post, image_dir, idx + 1, total, semaphore)
            for idx, post in enumerate(posts_with_images)
        ]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果，过滤异常
        valid_results = []
        success_count = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                post = posts_with_images[idx]
                print(f"  ⚠️  帖子 {post.mid} 分析失败: {result!r}", file=sys.stderr, flush=True)
                valid_results.append(
                    ImageAnalysisResult(
                        uid=post.uid,
                        mid=post.mid,
                        images=post.images[:self._config.max_images_per_post],
                        alignment_score=0.0,
                        summary=f"分析失败: {str(result)}",
                        error=str(result),
                    )
                )
            else:
                valid_results.append(result)
                if not result.error:
                    success_count += 1
        
        print(f"  ✅ 图文分析完成，成功处理 {success_count}/{total} 个帖子", flush=True)
        return valid_results

    async def _analyze_post_with_semaphore(
        self, post: WeiboPost, image_dir: str, idx: int, total: int, semaphore: asyncio.Semaphore
    ) -> ImageAnalysisResult:
        """使用信号量限制并发数的分析方法"""
        async with semaphore:
            result = await self._analyze_post_async(post, image_dir)
            print(f"  进度: {idx}/{total} - 帖子 {post.mid} 分析完成", flush=True)
            return result

    async def _analyze_post_async(self, post: WeiboPost, image_dir: str = "") -> ImageAnalysisResult:
        """异步分析单个帖子"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt_text = self._build_prompt(post, current_time)
        image_files = post.images[: self._config.max_images_per_post]

        # 异步加载所有图片（从本地文件或URL）
        image_tasks = [
            self._image_to_data_uri_async(image_ref, image_dir)
            for image_ref in image_files
        ]

        try:
            image_results = await asyncio.gather(*image_tasks, return_exceptions=True)

            # 处理图片加载结果，过滤异常
            image_data_uris: list[str] = []
            for image_result in image_results:
                if isinstance(image_result, Exception):
                    raise image_result
                image_data_uris.append(image_result)
        except Exception as exc:
            return ImageAnalysisResult(
                uid=post.uid,
                mid=post.mid,
                images=image_files,
                alignment_score=0.0,
                summary="图片加载或编码失败",
                error=str(exc),
            )

        # 调用 Gemini 多模态分析
        response_text = await self._client.generate_content(
            prompt_text=prompt_text,
            image_data_uris=image_data_uris,
            max_output_tokens=1200,
        )

        return ImageAnalysisResult(
            uid=post.uid,
            mid=post.mid,
            images=image_files,
            alignment_score=0.0,
            summary=response_text[:500] if response_text else "分析结果为空",
        )

    async def _image_to_data_uri_async(self, image_ref: str, image_dir: str = "") -> str:
        """将图片（本地文件或URL）转换为 data URI"""
        from pathlib import Path
        
        # 如果是本地文件路径
        if image_dir and not image_ref.startswith(('http://', 'https://')):
            filepath = Path(image_dir) / image_ref
            if filepath.exists():
                content = filepath.read_bytes()
                content_type = mimetypes.guess_type(str(filepath))[0] or "image/jpeg"
                encoded = base64.b64encode(content).decode("utf-8")
                return f"data:{content_type};base64,{encoded}"
        
        # 如果是URL，则下载
        async with httpx.AsyncClient(
            timeout=self._config.api_timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(image_ref)
        response.raise_for_status()
        content_type = response.headers.get("content-type") or mimetypes.guess_type(image_ref)[0] or "image/jpeg"
        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"

    @staticmethod
    def _build_prompt(post: WeiboPost, current_time: str) -> str:
        return (
            "你是微博账号异常检测中的图文融合分析助手。"
            "请按照以下步骤对博文进行分析：\n"
            "第一步：总结博文文本的主要内容。\n"
            "第二步：总结图片中的视觉内容。\n"
            "第三步：结合文本和图片内容，一起分析图文是否存在与社交机器人相关的异常特征，例如："
            "模板化营销、搬运内容、错配图片、明显的内容农场痕迹、机械化表达模式等。\n"
            "请以一整段连贯的文本输出你的分析，不要使用JSON格式或其他结构化格式。\n"
            f"当前系统时间：{current_time}\n"
            f"博文正文：{post.text}\n"
            f"发布时间：{post.created_at or '未知'}"
        )

