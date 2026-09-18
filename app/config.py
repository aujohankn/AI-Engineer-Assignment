from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Creative Variant Service"
    redis_url: str = "redis://localhost:6379/0"
    artifact_root: Path = Path("data/jobs")
    archive_root: Path = Path("archive")
    openai_api_key: str | None = Field(default=None, repr=False)
    image_provider: Literal["sdxl", "openai"] = "sdxl"
    evaluation_provider: Literal["local", "openai"] = "local"
    image_model: str = "gpt-image-2"
    evaluation_model: str = "gpt-5.5"
    sdxl_api_url: str = "http://sdxl:8001"
    sdxl_api_timeout_seconds: float = Field(default=600, ge=1, le=3600)
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sdxl_strength: float = Field(default=0.28, ge=0.05, le=0.95)
    sdxl_guidance_scale: float = Field(default=7.0, ge=0, le=20)
    sdxl_steps: int = Field(default=30, ge=1, le=100)
    sdxl_max_edge: int = Field(default=1024, ge=512, le=1536)
    sdxl_cpu_offload: bool = True
    local_min_change: float = Field(default=0.025, ge=0, le=1)
    local_min_preservation: float = Field(default=0.58, ge=0, le=1)
    max_variants: int = Field(default=4, ge=1, le=8)
    max_generation_attempts: int = Field(default=2, ge=1, le=4)
    job_ttl_seconds: int = Field(default=7 * 24 * 60 * 60, ge=300)
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, ge=1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
