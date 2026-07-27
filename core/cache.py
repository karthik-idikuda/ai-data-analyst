"""TTL + LRU answer cache.

Cache key = SHA-256 of (question, schema fingerprint, conversation tail, model).

Including the schema fingerprint is the important part: a cached answer can never
be replayed against a different dataset, which is the obvious way a naive
question-keyed cache goes wrong. The conversation tail is included because
"and for last quarter?" means different things in different contexts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .config import get_settings
from .observability import get_logger

log = get_logger(__name__)


def make_key(
    question: str,
    schema_fingerprint: str,
    *,
    history_tail: str = "",
    model: str = "",
    extra: dict[str, Any] | None = None,
) -> str:
    payload = json.dumps(
        {
            "q": " ".join(question.lower().split()),
            "schema": schema_fingerprint,
            "history": history_tail,
            "model": model,
            "extra": extra or {},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class CacheEntry:
    value: Any
    created_at: float
    hits: int = 0


class AnswerCache:
    """LRU + TTL cache.

    TTL semantics are explicit rather than truthiness-based: ``ttl_s=0`` expires
    entries immediately (useful in tests and for a "no reuse" deployment), and
    only ``ttl_s=None`` disables expiry. Treating 0 as "no expiry" is the kind of
    quiet ambiguity that turns into a stale-answer bug.
    """

    def __init__(self, max_entries: int | None = None, ttl_s: int | None = -1) -> None:
        settings = get_settings()
        self.max_entries = max_entries or settings.cache_max_entries
        self.ttl_s: int | None = settings.cache_ttl_s if ttl_s == -1 else ttl_s
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if not get_settings().cache_enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self.ttl_s is not None and time.time() - entry.created_at >= self.ttl_s:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            entry.hits += 1
            self.hits += 1
            log.info("cache.hit", key=key[:12], hits=entry.hits)
            return entry.value

    def set(self, key: str, value: Any) -> None:
        if not get_settings().cache_enabled:
            return
        with self._lock:
            self._store[key] = CacheEntry(value=value, created_at=time.time())
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                evicted, _ = self._store.popitem(last=False)
                log.info("cache.evicted", key=evicted[:12])

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "ttl_s": self.ttl_s,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


ANSWER_CACHE = AnswerCache()
