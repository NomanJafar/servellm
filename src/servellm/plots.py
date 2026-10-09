"""Plots for the sweep and mixed-length experiments. Writes PNGs, returns their paths."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series(sweep: dict, key: str, scale: float = 1.0):
    levels = sweep["levels"]
    return [lv["concurrency"] for lv in levels], [lv[key] * scale for lv in levels]


def _line_chart(sweeps: dict[str, dict], key: str, ylabel: str, title: str, out: Path, scale=1.0, log_y=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, sweep in sweeps.items():
        xs, ys = _series(sweep, key, scale)
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([lv["concurrency"] for lv in next(iter(sweeps.values()))["levels"]])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_sweep(sweeps: dict[str, dict], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        _line_chart(sweeps, "output_tokens_per_s", "output tokens / s (all requests)",
                    "Throughput vs concurrency", out_dir / "throughput_vs_concurrency.png"),
        _line_chart(sweeps, "itl_p50_s", "inter-token latency p50 (ms)",
                    "Per-request ITL vs concurrency", out_dir / "itl_vs_concurrency.png", scale=1000),
        _line_chart(sweeps, "ttft_p50_s", "time to first token p50 (ms)",
                    "TTFT vs concurrency", out_dir / "ttft_vs_concurrency.png", scale=1000, log_y=True),
    ]


def plot_mixed(data: dict, out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short, long = data["short_tokens"], data["long_tokens"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n, color in ((short, "tab:green"), (long, "tab:red")):
        ts = [f["total_s"] for f in data["finish_times_s"] if f["max_tokens"] == n]
        ax.scatter(ts, [n] * len(ts), color=color, label=f"{n}-token requests", s=40)
    ax.set_xlabel("completion time (s)")
    ax.set_ylabel("requested output tokens")
    ax.set_title("Short requests finish early despite sharing the batch with long ones")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "mixed_finish_times.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    tl = data["metrics_timeline"]
    ax.step([s["t"] for s in tl], [s.get("running", 0) for s in tl], where="post")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vllm:num_requests_running")
    ax.set_title("Batch occupancy over time (slots freed as short requests finish)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "mixed_running_timeline.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return [p1, p2]
