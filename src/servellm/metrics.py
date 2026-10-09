"""Scrape and reduce vLLM's Prometheus /metrics endpoint.

Only a handful of series matter for this project (see CONCEPTS.md section 7).
`select_watched` maps them to short stable names so the experiments and plots
do not depend on vLLM's exact metric names, which have changed across versions.
"""

from __future__ import annotations

import re
from collections import defaultdict

import httpx

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")

# short name -> candidate metric names, newest first
WATCHED: dict[str, tuple[str, ...]] = {
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "prompt_tokens_total": ("vllm:prompt_tokens_total",),
    "generation_tokens_total": ("vllm:generation_tokens_total",),
    "ttft_mean_s": ("vllm:time_to_first_token_seconds",),
    "itl_mean_s": ("vllm:time_per_output_token_seconds",),
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name -> value, labels dropped, same-name series summed."""
    out: defaultdict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            try:
                out[m.group(1)] += float(m.group(3))
            except ValueError:
                continue
    return dict(out)


def select_watched(metrics: dict[str, float]) -> dict[str, float]:
    snap: dict[str, float] = {}
    for short, candidates in WATCHED.items():
        for name in candidates:
            if short.endswith("_mean_s"):
                total, count = metrics.get(f"{name}_sum"), metrics.get(f"{name}_count")
                if total is not None and count:
                    snap[short] = total / count
                    break
            elif name in metrics:
                snap[short] = metrics[name]
                break
    return snap


def snapshot(base_url: str, timeout: float = 5.0) -> dict[str, float]:
    resp = httpx.get(f"{base_url}/metrics", timeout=timeout)
    resp.raise_for_status()
    return select_watched(parse_prometheus(resp.text))


async def snapshot_async(client: httpx.AsyncClient) -> dict[str, float]:
    resp = await client.get("/metrics", timeout=5.0)
    resp.raise_for_status()
    return select_watched(parse_prometheus(resp.text))
