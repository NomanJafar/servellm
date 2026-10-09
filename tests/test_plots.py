from servellm.plots import plot_mixed, plot_sweep


def _level(c, tps, ttft, itl):
    return {"concurrency": c, "output_tokens_per_s": tps, "ttft_p50_s": ttft, "ttft_p95_s": ttft * 1.5,
            "itl_p50_s": itl, "itl_p95_s": itl * 1.5, "errors": 0, "requests": c * 2,
            "metrics_timeline": [{"t": 0.0, "running": c, "kv_cache_usage": 0.1 * c}]}


def test_plot_sweep_writes_three_figures(tmp_path):
    on = {"levels": [_level(1, 90, 0.05, 0.011), _level(8, 600, 0.08, 0.013), _level(32, 1500, 0.2, 0.02)]}
    off = {"levels": [_level(1, 90, 0.05, 0.011), _level(8, 95, 0.9, 0.011)]}
    paths = plot_sweep({"batching on": on, "batching off (max-num-seqs=1)": off}, tmp_path)
    assert sorted(p.name for p in paths) == ["itl_vs_concurrency.png", "throughput_vs_concurrency.png",
                                             "ttft_vs_concurrency.png"]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_plot_mixed_writes_timeline_and_finish_times(tmp_path):
    data = {
        "short_tokens": 16, "long_tokens": 512,
        "finish_times_s": [{"max_tokens": 16, "total_s": 0.4}, {"max_tokens": 512, "total_s": 9.0}],
        "metrics_timeline": [{"t": 0.0, "running": 2.0}, {"t": 0.5, "running": 1.0}, {"t": 9.0, "running": 0.0}],
    }
    paths = plot_mixed(data, tmp_path)
    assert sorted(p.name for p in paths) == ["mixed_finish_times.png", "mixed_running_timeline.png"]
