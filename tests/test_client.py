import pytest

from servellm.client import Chunk, parse_sse_line, summarize


def _data(obj):
    import json
    return "data: " + json.dumps(obj)


def test_parse_sse_line_returns_delta_text():
    line = _data({"choices": [{"delta": {"content": "Hel"}}]})
    assert parse_sse_line(line) == {"text": "Hel", "usage": None}


def test_parse_sse_line_returns_usage_from_final_chunk():
    line = _data({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 40}})
    assert parse_sse_line(line) == {"text": "", "usage": {"prompt_tokens": 12, "completion_tokens": 40}}


@pytest.mark.parametrize("line", ["", "data: [DONE]", ": keep-alive", "event: ping"])
def test_parse_sse_line_ignores_non_data_lines(line):
    assert parse_sse_line(line) is None


def test_summarize_computes_ttft_itl_and_throughput():
    t0 = 100.0
    chunks = [
        Chunk(t=100.0, text="", usage=None),        # role-only first chunk: no content, not TTFT
        Chunk(t=100.25, text="A", usage=None),      # first content token -> TTFT 0.25
        Chunk(t=100.27, text="B", usage=None),      # ITL 0.02
        Chunk(t=100.31, text="C", usage=None),      # ITL 0.04
        Chunk(t=100.32, text="", usage={"prompt_tokens": 7, "completion_tokens": 3}),
    ]
    r = summarize(t0, chunks)
    assert r.ttft_s == pytest.approx(0.25)
    assert r.itls_s == pytest.approx([0.02, 0.04])
    assert r.total_s == pytest.approx(0.32)
    assert r.prompt_tokens == 7
    assert r.completion_tokens == 3
    assert r.text == "ABC"
    assert r.decode_tokens_per_s == pytest.approx(2 / 0.06)


def test_summarize_falls_back_to_counting_chunks_without_usage():
    chunks = [Chunk(t=1.0, text="x", usage=None), Chunk(t=1.1, text="y", usage=None)]
    r = summarize(0.5, chunks)
    assert r.completion_tokens == 2
    assert r.prompt_tokens is None


def test_summarize_with_no_content_has_no_ttft():
    r = summarize(0.0, [Chunk(t=0.1, text="", usage=None)])
    assert r.ttft_s is None
    assert r.itls_s == []
