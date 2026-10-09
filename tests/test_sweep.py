import asyncio
import json

import httpx
import pytest

from servellm.client import StreamResult
from servellm.sweep import aggregate, make_prompt, percentile, run_level


def test_percentile_interpolates_and_handles_edges():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 4
    assert percentile([], 50) is None


def _res(ttft, itls, completion, error=None):
    return StreamResult(
        ttft_s=ttft, itls_s=itls, total_s=(ttft or 0) + sum(itls), prompt_tokens=10,
        completion_tokens=completion, text="x" * completion, error=error,
    )


def test_aggregate_reports_latency_percentiles_and_throughput():
    results = [_res(0.1, [0.02, 0.02], 3), _res(0.3, [0.04, 0.04], 3)]
    agg = aggregate(results, wall_s=2.0)
    assert agg["requests"] == 2
    assert agg["errors"] == 0
    assert agg["ttft_p50_s"] == pytest.approx(0.2)
    assert agg["ttft_p95_s"] == pytest.approx(0.29)
    assert agg["itl_p50_s"] == pytest.approx(0.03)
    assert agg["completion_tokens"] == 6
    assert agg["output_tokens_per_s"] == pytest.approx(3.0)  # 6 tokens / 2 s wall


def test_aggregate_excludes_errored_requests_from_latency():
    results = [_res(0.1, [0.02], 2), _res(None, [], 0, error="ReadTimeout")]
    agg = aggregate(results, wall_s=1.0)
    assert agg["errors"] == 1
    assert agg["ttft_p50_s"] == pytest.approx(0.1)


def test_make_prompt_scales_with_requested_tokens():
    short, long = make_prompt(50), make_prompt(500)
    assert 40 <= len(short.split()) <= 60
    assert len(long) > 5 * len(short)


class FakeVllm:
    """ASGI-free fake: answers chat completions with N streamed tokens, tracks concurrency."""

    def __init__(self, tokens=3, delay=0.01):
        self.tokens, self.delay = tokens, delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.seen_bodies = []

    async def _stream(self):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            for i in range(self.tokens):
                await asyncio.sleep(self.delay)
                yield f'data: {{"choices":[{{"delta":{{"content":"t{i} "}}}}]}}\n\n'.encode()
            usage = {"prompt_tokens": 5, "completion_tokens": self.tokens}
            yield f'data: {{"choices":[],"usage":{json.dumps(usage)}}}\n\n'.encode()
            yield b"data: [DONE]\n\n"
        finally:
            self.in_flight -= 1

    async def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        self.seen_bodies.append(json.loads(request.content))
        return httpx.Response(200, stream=_AsyncByteStream(self._stream()))


class _AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, agen):
        self._agen = agen

    async def __aiter__(self):
        async for chunk in self._agen:
            yield chunk


def test_run_level_keeps_requested_concurrency_and_returns_all_results():
    fake = FakeVllm(tokens=4)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="http://fake") as c:
            return await run_level(c, model="m", concurrency=4, rounds=2, prompt="hi", max_tokens=4)

    level = asyncio.run(go())
    assert level["concurrency"] == 4
    assert level["requests"] == 8
    assert level["errors"] == 0
    assert level["completion_tokens"] == 32
    assert fake.max_in_flight == 4
    assert all(b["max_tokens"] == 4 and b["stream"] is True for b in fake.seen_bodies)
    assert level["ttft_p50_s"] > 0
