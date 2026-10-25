"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def stable_json(value: Any) -> str:
    """Serialise ``value`` to canonical JSON (sorted keys, no whitespace).

    Canonical JSON makes hashes and exports byte-identical across runs.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(*parts: Any) -> str:
    """Build a SHA-256 cache key from an ordered sequence of parts."""
    return hashlib.sha256(stable_json(list(parts)).encode("utf-8")).hexdigest()
