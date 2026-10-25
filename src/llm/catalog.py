"""Load the provider/model catalogue from ``config/providers.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelOption(BaseModel):
    """One selectable model."""

    id: str
    label: str


class ProviderOption(BaseModel):
    """One selectable provider."""

    id: str
    label: str
    secret_name: str | None = None
    requires_key: bool = True
    supports_temperature: bool = True
    models: list[ModelOption]


def load_catalog(path: Path) -> list[ProviderOption]:
    """Parse the YAML catalogue, preserving file order."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [ProviderOption(id=pid, **spec) for pid, spec in (raw.get("providers") or {}).items()]
