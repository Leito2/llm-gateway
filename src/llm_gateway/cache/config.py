"""config/cache.yaml: thresholds per embedder and cache tuning."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Thresholds(BaseModel):
    """`t_high`: serve without the verifier. `t_low`: below it, miss. Between them the verifier decides.
    `null` means "not calibrated with G6 yet": the layer observes and never serves (PLAN §6.3 point 5)."""

    t_high: float | None = Field(default=None, gt=0, le=1)
    t_low: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Thresholds":
        if self.t_high is None:
            if self.t_low is not None:
                raise ValueError("t_low without t_high")
            return self
        if self.t_low is None:
            self.t_low = self.t_high  # no gray zone
        if self.t_low > self.t_high:
            raise ValueError("t_low must not exceed t_high")
        return self

    @property
    def calibrated(self) -> bool:
        return self.t_high is not None


class CacheConfig(BaseModel):
    max_entries: int = Field(default=5000, gt=0)  # memory backend
    stream_pace_ms: float = Field(default=0.0, ge=0)  # delay between chunks when a hit is streamed
    embed_timeout_s: float = Field(default=0.3, gt=0)  # PLAN §4.1
    verify_timeout_s: float = Field(default=0.5, gt=0)
    write_queue: int = Field(default=256, gt=0)  # pending write-backs; more are dropped, not awaited
    embedders: dict[str, Thresholds] = Field(default_factory=dict)

    def thresholds(self, embedder: str) -> Thresholds:
        return self.embedders.get(embedder, Thresholds())


def load_cache_config(path: Path) -> CacheConfig:
    if not path.exists():
        return CacheConfig()
    return CacheConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
