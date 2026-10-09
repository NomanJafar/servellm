from servellm.metrics import WATCHED, parse_prometheus, select_watched

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 12.0
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 3.0
vllm:kv_cache_usage_perc{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 0.4375
vllm:prompt_tokens_total{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 48210.0
vllm:generation_tokens_total{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 120034.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="x"} 5.0
vllm:time_to_first_token_seconds_sum{model_name="x"} 9.5
vllm:time_to_first_token_seconds_count{model_name="x"} 20.0
"""


def test_parse_prometheus_strips_labels_and_skips_comments():
    m = parse_prometheus(SAMPLE)
    assert m["vllm:num_requests_running"] == 12.0
    assert m["vllm:kv_cache_usage_perc"] == 0.4375
    assert "# HELP" not in "".join(m)


def test_parse_prometheus_sums_series_with_different_labels():
    text = 'g{a="1"} 2\ng{a="2"} 3\n'
    assert parse_prometheus(text)["g"] == 5.0


def test_select_watched_returns_only_known_gauges_and_counters():
    snap = select_watched(parse_prometheus(SAMPLE))
    assert set(snap) <= set(WATCHED)
    assert snap["running"] == 12.0
    assert snap["waiting"] == 3.0
    assert snap["kv_cache_usage"] == 0.4375
    assert snap["generation_tokens_total"] == 120034.0
    assert snap["ttft_mean_s"] == 9.5 / 20


def test_select_watched_falls_back_to_old_cache_metric_name():
    old = 'vllm:gpu_cache_usage_perc{m="x"} 0.25\n'
    assert select_watched(parse_prometheus(old))["kv_cache_usage"] == 0.25
