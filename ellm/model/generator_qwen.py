"""The real generator — Qwen2.5-1.5B inside the materialized context.

Implements the `Generator` protocol: materialize-to-prompt gives `tid{n}`
tokens, Qwen generates a continuation, and the `tid{n}` ids in that
continuation become the next turn.
"""

from __future__ import annotations

import re

from .base import QwenHub

_TID_RE = re.compile(r"tid(\d+)")


class QwenGenerator:
    """Deterministic generation (greedy) over the materialized context."""

    def __init__(self, hub: QwenHub, max_new_tokens: int = 16):
        self.hub = hub
        self.max_new_tokens = max_new_tokens

    def generate(self, context_tokens: list[str], policy: dict | None = None) -> list[int]:
        prompt = " ".join(context_tokens) if context_tokens else "tid1"
        text = self.hub.generate(prompt, max_new_tokens=self.max_new_tokens)
        ids = [int(m) for m in _TID_RE.findall(text)]
        if not ids:
            # Fallback: keep the loop closed deterministically.
            ids = [int(t[3:]) for t in context_tokens if t.startswith("tid")] or [1]
            if ids:
                ids = [*ids, max(ids) + 1]
        return ids
