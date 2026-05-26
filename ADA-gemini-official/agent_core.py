from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from datetime import datetime
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from config import AppConfig
from models import (
    BotDetectionResult,
    CrawlResult,
    ImageAnalysisResult,
    ToolTrace,
    WeiboPost,
)
from multimodal_checker import MultimodalConsistencyChecker
from weibo_scraper import WeiboScraper


# ── Structured output schema ────────────────────────────────────────


class BotVerdict(BaseModel):
    """机器人检测最终判定结果"""

    verdict: str = Field(description="判定结果，必须是 likely_bot 或 likely_human")
    confidence: float = Field(description="置信度，0 到 1 之间的小数")
    summary: str = Field(description="详细分析推理过程")


# ── Shared run-state ────────────────────────────────────────────────


def _make_run_state() -> dict[str, Any]:
    return {
        "account_input": "",
        "crawl_result": None,
        "image_analyses": [],
        "heuristics": None,
        "traces": [],
    }


# ── Tool factories ──────────────────────────────────────────────────


def _make_tools(
    scraper: WeiboScraper,
    multimodal_checker: MultimodalConsistencyChecker,
    config: AppConfig,
    state: dict[str, Any],
    output_dir: str | None = None,
):
    """返回供 deep agent 使用的 tool 函数列表。"""

    effective_output_dir = output_dir if output_dir is not None else config.output_dir

    def scrape_weibo_account(account_input: str) -> str:
        """抓取微博账号的主页资料和最近博文。输入为微博 uid 或完整主页 URL。
        返回抓取到的账号摘要信息。必须首先调用此工具获取原始数据。"""
        crawl_result: CrawlResult = scraper.scrape_account(
            account_input, output_dir=effective_output_dir
        )

        # 按参考日期过滤超期博文
        original_count = len(crawl_result.posts)
        if config.reference_date:
            cutoff = datetime.strptime(config.reference_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59
            )

            def _within_cutoff(p: WeiboPost) -> bool:
                if not p.created_at:
                    return True  # 时间未知，保留
                try:
                    return datetime.strptime(p.created_at, "%Y-%m-%d %H:%M:%S") <= cutoff
                except ValueError:
                    return True  # 无法解析，保留

            crawl_result.posts = [p for p in crawl_result.posts if _within_cutoff(p)]

        state["account_input"] = account_input
        state["crawl_result"] = crawl_result
        filtered_count = len(crawl_result.posts)
        filter_note = (
            f"（参考日期 {config.reference_date}：过滤前 {original_count} 条→过滤后 {filtered_count} 条）"
            if config.reference_date and filtered_count != original_count
            else ""
        )
        state["traces"].append(
            ToolTrace(
                tool_name="scrape_weibo_account",
                status="success",
                summary=f"抓取到 {filtered_count} 条博文{filter_note}",
                metadata={"uid": crawl_result.profile.uid},
            )
        )
        profile = crawl_result.profile
        post_summaries = []
        for i, p in enumerate(crawl_result.posts[:20]):
            has_img = "有图" if p.images else "纯文"
            post_summaries.append(
                f"  {i+1}. [{has_img}] {p.created_at or '未知时间'} | {p.text[:80]}"
            )
        return (
            f"抓取完成。\n"
            f"账号: {profile.screen_name} (uid={profile.uid})\n"
            f"简介: {profile.description[:100]}\n"
            f"粉丝: {profile.followers_count}, 关注: {profile.friends_count}, "
            f"博文总数: {profile.statuses_count}\n"
            f"注册时间: {profile.created_at}\n"
            f"已抓取 {len(crawl_result.posts)} 条博文:\n"
            + "\n".join(post_summaries)
        )

    def analyze_image_posts() -> str:
        """对所有含图片的博文进行图文一致性多模态分析。
        会检测图文是否匹配、是否有营销/搬运/内容农场等异常特征。
        必须在 scrape_weibo_account 之后调用。"""
        crawl = state.get("crawl_result")
        if crawl is None:
            return "错误：尚未抓取账号数据，请先调用 scrape_weibo_account。"

        image_analyses: list[ImageAnalysisResult] = asyncio.run(
            multimodal_checker.analyze_posts(crawl.posts, image_dir=crawl.image_dir)
        )
        state["image_analyses"] = image_analyses
        state["traces"].append(
            ToolTrace(
                tool_name="analyze_image_posts",
                status="success",
                summary=f"完成 {len(image_analyses)} 条图文一致性分析",
                metadata={"posts_with_images": len(image_analyses)},
            )
        )
        if not image_analyses:
            return "该账号没有含图片的博文，无需图文分析。"
        lines = []
        for item in image_analyses:
            status = "✅" if not item.error else "❌"
            lines.append(f"  {status} mid={item.mid}: {item.summary[:120]}")
        return (
            f"图文分析完成，共分析 {len(image_analyses)} 条帖子:\n"
            + "\n".join(lines)
        )

    def compute_heuristics() -> str:
        """根据已抓取的博文数据计算统计特征（重复率、短文比例、平均图文一致性等）。
        必须在 scrape_weibo_account 之后调用。图文分析如已完成会纳入统计。"""
        crawl = state.get("crawl_result")
        if crawl is None:
            return "错误：尚未抓取账号数据，请先调用 scrape_weibo_account。"

        posts = crawl.posts
        image_analyses = state.get("image_analyses", [])
        heuristics = _build_heuristic_summary(posts, image_analyses)
        state["heuristics"] = heuristics
        state["traces"].append(
            ToolTrace(
                tool_name="compute_heuristics",
                status="success",
                summary="统计特征计算完成",
                metadata=heuristics,
            )
        )

        hours: list[int] = []
        for p in posts:
            if p.created_at:
                try:
                    dt = datetime.strptime(p.created_at, "%Y-%m-%d %H:%M:%S")
                    hours.append(dt.hour)
                except ValueError:
                    pass
        hour_dist = ""
        if hours:
            hc = Counter(hours)
            top3 = hc.most_common(3)
            hour_dist = f"发帖高峰时段: {', '.join(f'{h}时({c}条)' for h, c in top3)}"

        return (
            f"统计特征:\n"
            f"  博文总数: {heuristics['total_posts']}\n"
            f"  图文分析数: {heuristics['posts_with_image_analysis']}\n"
            f"  文本重复率: {heuristics['duplicate_ratio']}\n"
            f"  平均图文一致性: {heuristics['average_alignment']}\n"
            f"  短文比例(≤25字): {heuristics['short_text_ratio']}\n"
            f"  {hour_dist}"
        )

    return [scrape_weibo_account, analyze_image_posts, compute_heuristics]


# ── Heuristic helper ────────────────────────────────────────────────


def _build_heuristic_summary(
    posts: list[WeiboPost], image_analyses: list[ImageAnalysisResult]
) -> dict[str, Any]:
    texts = [post.text.strip() for post in posts if post.text.strip()]
    short_texts = [text for text in texts if len(text) <= 25]
    duplicate_ratio = 0.0
    if texts:
        counts = Counter(texts)
        duplicate_ratio = sum(count - 1 for count in counts.values() if count > 1) / len(texts)
    alignment_scores = [item.alignment_score for item in image_analyses if not item.error]
    avg_alignment = mean(alignment_scores) if alignment_scores else 0.0

    return {
        "total_posts": len(posts),
        "posts_with_image_analysis": len(image_analyses),
        "duplicate_ratio": round(duplicate_ratio, 3),
        "average_alignment": round(avg_alignment, 3),
        "short_text_ratio": round(len(short_texts) / len(texts), 3) if texts else 0.0,
    }


# ── System prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个专业的微博机器人账号检测智能体。你的任务是判断给定的微博账号更可能是机器人(likely_bot)还是真人(likely_human)。

你有以下工具可用：
1. **scrape_weibo_account** — 抓取微博账号主页资料和最近博文，这是必须首先执行的步骤。
2. **analyze_image_posts** — 对含图片的博文做多模态图文一致性分析，检测搬运、营销等异常。
3. **compute_heuristics** — 计算博文的统计特征（重复率、短文比例、发帖时间分布等）。

## 工作方式
- 你可以自主决定调用哪些工具、以什么顺序调用
- 在收集到足够信息后，给出最终的判定结论
- 如果账号博文全是纯文字没有图片，可以跳过图文分析
- 你应该综合多个维度的信息来做判断，不要仅凭单一指标

## 判断维度参考（不限于此）
- 账号资料完整度（头像、简介、认证状态）
- 博文内容多样性 vs 重复/模板化
- 图文一致性（是否搬运、错配图片）
- 发帖时间模式（是否机械化）
- 互动数据（转评赞比例是否合理）
- 内容质量（是否有营销、内容农场痕迹）

## 重要提示
{time_context}
- verdict 只能是 likely_bot 或 likely_human，不能输出 uncertain
- 如果账户内容声明了自己是投稿bot或者bot的，都是bot
- 判断一个账号是否是机器人和人类的核心依据是是否存在真实人类活动/自动化控制的痕迹。很多账号因为工作、生活等原因，会发布一些机械化的博文(如广告、搬运、投稿)，但其他博文可能存在人类活动的痕迹。
"""


# ── Main agent class ────────────────────────────────────────────────


class WeiboBotDetectorAgent:
    def __init__(
        self,
        *,
        config: AppConfig,
        scraper: WeiboScraper,
        multimodal_checker: MultimodalConsistencyChecker,
    ) -> None:
        self._config = config
        self._scraper = scraper
        self._multimodal_checker = multimodal_checker

    def run(self, account_input: str, output_dir: str | None = None) -> BotDetectionResult:
        from deepagents import create_deep_agent
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI

        load_dotenv()

        run_state = _make_run_state()
        tools = _make_tools(
            self._scraper, self._multimodal_checker, self._config, run_state,
            output_dir=output_dir,
        )

        if self._config.reference_date:
            time_context = (
                f"数据截止日期：{self._config.reference_date}（所有晚于该日期的博文已被过滤，请基于此日期视角进行分析）"
            )
        else:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_context = (
                f"当前系统时间：{current_time}\n"
                f"- 不要仅因为年份是 2026 就认定为“未来时间”，必须与当前系统时间比较"
            )
        system_prompt = SYSTEM_PROMPT.format(time_context=time_context)

        gemini_llm = ChatGoogleGenerativeAI(
            model=self._config.gemini_model,
            google_api_key=self._config.gemini_api_key,
            temperature=0,
        )

        agent = create_deep_agent(
            model=gemini_llm,
            tools=tools,
            system_prompt=system_prompt,
            response_format=BotVerdict,
        )

        print(f"[Agent] 开始检测账号: {account_input}", flush=True)
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"请检测以下微博账号是否为机器人: {account_input}",
                    }
                ]
            }
        )

        # 从 structured_response 提取判定
        verdict_obj: BotVerdict | None = result.get("structured_response")

        if verdict_obj is None:
            # fallback: 从最后一条消息解析
            last_msg = result["messages"][-1].content
            parsed = _parse_json_block(
                last_msg if isinstance(last_msg, str) else str(last_msg)
            )
            verdict_str = str(parsed.get("verdict", "likely_human"))
            confidence = float(parsed.get("confidence", 0.5))
            summary = str(
                parsed.get(
                    "summary",
                    last_msg[:500] if isinstance(last_msg, str) else str(last_msg)[:500],
                )
            )
        else:
            verdict_str = verdict_obj.verdict
            confidence = verdict_obj.confidence
            summary = verdict_obj.summary

        run_state["traces"].append(
            ToolTrace(
                tool_name="agent_final_verdict",
                status="success",
                summary=summary[:200],
                metadata={
                    "model": self._config.gemini_model
                },
            )
        )

        crawl = run_state.get("crawl_result")
        if crawl is None:
            raise RuntimeError("Agent 未调用 scrape_weibo_account，无法生成结果")

        return BotDetectionResult(
            account_input=account_input,
            profile=crawl.profile,
            posts=crawl.posts,
            image_analyses=run_state.get("image_analyses", []),
            verdict=verdict_str,
            confidence=confidence,
            summary=summary,
            traces=run_state["traces"],
        )


def _parse_json_block(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
