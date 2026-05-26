from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class UserProfile:
    screen_name: str = ""
    description: str = ""
    verified: bool = False
    default_profile_image: bool = False
    created_at: str = ""
    location: str = ""
    followers_count: int = 0
    friends_count: int = 0
    statuses_count: int = 0
    interactions_count: int = 0
    uid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WeiboPost:
    uid: str
    mid: str
    created_at: str
    text: str
    is_retweet: bool = False
    parent_mid: str | None = None
    parent_text: str | None = None
    reposts_count: int = 0
    comments_count: int = 0
    attitudes_count: int = 0
    images: list[str] = field(default_factory=list)
    video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CrawlResult:
    profile: UserProfile
    posts: list[WeiboPost]
    image_dir: str = ""  # 存储图片的目录路径
    crawled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "posts": [post.to_dict() for post in self.posts],
            "crawled_at": self.crawled_at,
        }


@dataclass(slots=True)
class ImageAnalysisResult:
    uid: str
    mid: str
    images: list[str]
    alignment_score: float
    summary: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolTrace:
    tool_name: str
    status: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BotDetectionResult:
    account_input: str
    profile: UserProfile
    posts: list[WeiboPost]
    image_analyses: list[ImageAnalysisResult]
    verdict: str
    confidence: float
    summary: str
    traces: list[ToolTrace] = field(default_factory=list)
    raw_data_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_input": self.account_input,
            "profile": self.profile.to_dict(),
            "posts": [post.to_dict() for post in self.posts],
            "image_analyses": [item.to_dict() for item in self.image_analyses],
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "traces": [trace.to_dict() for trace in self.traces],
            "raw_data_paths": self.raw_data_paths,
        }
