"""Minimal streaming client for /v1/chat/completions that records per-token timing.

Deliberately small: enough to measure TTFT and inter-token latency for the
experiments in this project. The full harness lives in the `tokenbench` project.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import pairwise

import httpx


@dataclass(frozen=True)
class Chunk:
    t: float                 # monotonic time the chunk arrived
    text: str                # delta content ("" for role-only or usage-only chunks)
    usage: dict | None       # present only on the final chunk when include_usage is set


@dataclass
class StreamResult:
    ttft_s: float | None
    itls_s: list[float]
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int
    text: str
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Tokens per second during decode only (excludes prefill/TTFT)."""
        decode_time = sum(self.itls_s)
        return len(self.itls_s) / decode_time if decode_time > 0 else None

    @property
    def mean_itl_s(self) -> float | None:
        return sum(self.itls_s) / len(self.itls_s) if self.itls_s else None


def parse_sse_line(line: str) -> dict | None:
    """Return {"text", "usage"} for a data line, None for anything else."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if not payload or payload == "[DONE]":
        return None
    obj = json.loads(payload)
    text = ""
    for choice in obj.get("choices", []):
        text += choice.get("delta", {}).get("content") or ""
    return {"text": text, "usage": obj.get("usage")}


def summarize(t_start: float, chunks: list[Chunk]) -> StreamResult:
    content = [c for c in chunks if c.text]
    ttft = content[0].t - t_start if content else None
    itls = [b.t - a.t for a, b in pairwise(content)]
    total = (chunks[-1].t - t_start) if chunks else 0.0
    usage = next((c.usage for c in reversed(chunks) if c.usage), None)
    return StreamResult(
        ttft_s=ttft,
        itls_s=itls,
        total_s=total,
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
        completion_tokens=usage.get("completion_tokens") if usage else len(content),
        text="".join(c.text for c in content),
    )


async def stream_chat(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    seed: int | None = 0,
    ignore_eos: bool = True,
) -> StreamResult:
    """One streaming request; returns timing for it.

    `ignore_eos` is a vLLM extension that forces exactly `max_tokens` output tokens so
    that runs are comparable; turn it off for realistic length variation.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    if ignore_eos:
        body["ignore_eos"] = True

    chunks: list[Chunk] = []
    t_start = time.monotonic()
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body, timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                parsed = parse_sse_line(line)
                if parsed is not None:
                    chunks.append(Chunk(t=time.monotonic(), **parsed))
    except httpx.HTTPError as e:
        result = summarize(t_start, chunks)
        result.error = f"{type(e).__name__}: {e}"
        return result
    return summarize(t_start, chunks)
