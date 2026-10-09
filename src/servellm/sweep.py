"""Experiments: concurrency sweep, batching on/off, mixed lengths.

Each experiment is a closed-loop load generator: `concurrency` workers each
send a request, wait for it to finish, then send the next, until `rounds`
requests per worker are done. Closed loop means the offered load adapts to the
server; it is the right model for "N users typing as fast as answers arrive"
and keeps queueing out of the picture so TTFT/ITL reflect the batch size.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from servellm.client import StreamResult, stream_chat
from servellm.metrics import snapshot_async

_FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineer measures "
    "latency throughput memory and cost under steadily increasing load "
)


def make_prompt(approx_tokens: int) -> str:
    """Roughly `approx_tokens` tokens of English filler (about one token per word)."""
    words = _FILLER.split()
    return " ".join(words[i % len(words)] for i in range(approx_tokens))


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def aggregate(results: list[StreamResult], wall_s: float) -> dict:
    ok = [r for r in results if r.error is None and r.ttft_s is not None]
    ttfts = [r.ttft_s for r in ok]
    itls = [itl for r in ok for itl in r.itls_s]
    completion = sum(r.completion_tokens for r in ok)
    return {
        "requests": len(results),
        "errors": sum(1 for r in results if r.error is not None),
        "wall_s": wall_s,
        "completion_tokens": completion,
        "output_tokens_per_s": completion / wall_s if wall_s > 0 else None,
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_mean_s": sum(ttfts) / len(ttfts) if ttfts else None,
        "itl_p50_s": percentile(itls, 50),
        "itl_p95_s": percentile(itls, 95),
        "itl_mean_s": sum(itls) / len(itls) if itls else None,
    }


async def _worker(client, model, prompt, max_tokens, rounds, out: list[StreamResult]):
    for _ in range(rounds):
        out.append(await stream_chat(client, model, [{"role": "user", "content": prompt}], max_tokens))


async def _sample_metrics(client, interval_s: float, out: list[dict], stop: asyncio.Event):
    t0 = time.monotonic()
    while not stop.is_set():
        try:
            snap = await snapshot_async(client)
            out.append({"t": time.monotonic() - t0, **snap})
        except (httpx.HTTPError, AssertionError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), interval_s)
        except TimeoutError:
            pass


async def run_level(
    client: httpx.AsyncClient,
    model: str,
    concurrency: int,
    rounds: int,
    prompt: str,
    max_tokens: int,
    sample_metrics: bool = False,
    metrics_interval_s: float = 0.5,
) -> dict:
    results: list[StreamResult] = []
    timeline: list[dict] = []
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_metrics(client, metrics_interval_s, timeline, stop)) if sample_metrics else None

    t0 = time.monotonic()
    await asyncio.gather(*[
        _worker(client, model, prompt, max_tokens, rounds, results) for _ in range(concurrency)
    ])
    wall = time.monotonic() - t0
    stop.set()
    if sampler:
        await sampler

    level = {"concurrency": concurrency, "max_tokens": max_tokens, **aggregate(results, wall)}
    level["metrics_timeline"] = timeline
    level["per_request"] = [
        {"ttft_s": r.ttft_s, "mean_itl_s": r.mean_itl_s, "total_s": r.total_s,
         "completion_tokens": r.completion_tokens, "error": r.error}
        for r in results
    ]
    return level


async def concurrency_sweep(
    base_url: str, model: str, levels: list[int], prompt_tokens: int, max_tokens: int, rounds: int
) -> dict:
    prompt = make_prompt(prompt_tokens)
    out = {"model": model, "prompt_tokens_target": prompt_tokens, "max_tokens": max_tokens,
           "rounds": rounds, "levels": []}
    async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
        # warm-up so the first level does not pay for CUDA graph capture / cold caches
        await stream_chat(client, model, [{"role": "user", "content": prompt}], max_tokens)
        for c in levels:
            level = await run_level(client, model, c, rounds, prompt, max_tokens, sample_metrics=True)
            out["levels"].append(level)
            print(f"concurrency={c:>3}  tok/s={level['output_tokens_per_s']:.1f}  "
                  f"ttft_p50={level['ttft_p50_s']*1000:.0f}ms  itl_p50={level['itl_p50_s']*1000:.1f}ms")
    return out


async def mixed_lengths(base_url: str, model: str, short_tokens: int, long_tokens: int, n_each: int) -> dict:
    """Launch short and long requests together; record when each finishes.

    With continuous batching, short requests should finish at roughly
    short_tokens x ITL regardless of the long requests sharing the batch.
    """
    prompt = make_prompt(64)
    async with httpx.AsyncClient(base_url=base_url, timeout=None) as client:
        timeline: list[dict] = []
        stop = asyncio.Event()
        sampler = asyncio.create_task(_sample_metrics(client, 0.25, timeline, stop))
        t0 = time.monotonic()
        tasks = [stream_chat(client, model, [{"role": "user", "content": prompt}], n)
                 for n in [short_tokens] * n_each + [long_tokens] * n_each]
        results = await asyncio.gather(*tasks)
        stop.set()
        await sampler
    return {
        "short_tokens": short_tokens, "long_tokens": long_tokens, "n_each": n_each,
        "finish_times_s": [{"max_tokens": n, "total_s": r.total_s, "completion_tokens": r.completion_tokens}
                           for n, r in zip([short_tokens] * n_each + [long_tokens] * n_each, results, strict=True)],
        "metrics_timeline": timeline,
        "wall_s": time.monotonic() - t0,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="servellm experiments")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", type=Path, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--levels", default="1,2,4,8,16,32,64")
    s.add_argument("--prompt-tokens", type=int, default=256)
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--rounds", type=int, default=2)
    m = sub.add_parser("mixed")
    m.add_argument("--short", type=int, default=16)
    m.add_argument("--long", type=int, default=512)
    m.add_argument("--n-each", type=int, default=8)
    a = p.parse_args(argv)

    if a.cmd == "sweep":
        data = asyncio.run(concurrency_sweep(
            a.base_url, a.model, [int(x) for x in a.levels.split(",")], a.prompt_tokens, a.max_tokens, a.rounds))
    else:
        data = asyncio.run(mixed_lengths(a.base_url, a.model, a.short, a.long, a.n_each))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
