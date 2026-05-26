from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    load_dotenv = None


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    llm_base_url: str
    llm_api_key: str = ""
    gemini_api_key: str = ""
    llm_model: str = "qwen3.5-flash"
    gemini_model: str = "gemini-2.0-flash"
    multimodal_model: str | None = None
    final_reasoner_model: str | None = None
    weibo_cookies_path: str = ".auth/cookies.json"
    output_dir: str = "outputs"
    max_posts: int = 20
    max_images_per_post: int = 4
    headless: bool = True
    page_timeout_ms: int = 45000
    api_timeout_seconds: float = 180.0
    user_agent: str = DEFAULT_USER_AGENT
    reference_date: str | None = None  # 数据截止日期 YYYY-MM-DD，None 表示不过滤

    @classmethod
    def from_env(cls) -> "AppConfig":
        if load_dotenv is not None:
            load_dotenv()

        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "https://www.dmxapi.cn"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", "qwen3.5-flash"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            multimodal_model=os.getenv("MULTIMODAL_MODEL") or None,
            final_reasoner_model=os.getenv("FINAL_REASONER_MODEL") or None,
            output_dir=os.getenv("OUTPUT_DIR", "outputs"),
            max_posts=int(os.getenv("MAX_POSTS", "20")),
            max_images_per_post=int(os.getenv("MAX_IMAGES_PER_POST", "4")),
            headless=_as_bool(os.getenv("WEIBO_HEADLESS"), True),
            page_timeout_ms=int(os.getenv("PAGE_TIMEOUT_MS", "45000")),
            api_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "180")),
            user_agent=os.getenv("WEIBO_USER_AGENT", DEFAULT_USER_AGENT),
            reference_date=os.getenv("REFERENCE_DATE") or None,
        )

    @property
    def effective_multimodal_model(self) -> str:
        return self.multimodal_model or self.llm_model

    @property
    def effective_reasoner_model(self) -> str:
        return self.final_reasoner_model or self.llm_model

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent

    def resolve_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def validate(self) -> None:
        if not self.llm_base_url.strip():
            raise ValueError("LLM_BASE_URL 不能为空")
        if not self.gemini_api_key.strip():
            raise ValueError("GEMINI_API_KEY 不能为空")
        if self.max_posts <= 0:
            raise ValueError("MAX_POSTS 必须大于 0")
        if self.max_posts > 20:
            raise ValueError("MAX_POSTS 不能超过 20")
        if self.max_images_per_post <= 0:
            raise ValueError("MAX_IMAGES_PER_POST 必须大于 0")
