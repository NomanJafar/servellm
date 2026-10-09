"""Build, start and inspect a vLLM server process.

The interesting part for learning is `parse_startup_log`: vLLM prints, once at
startup, how much memory it gave to the KV cache, how many tokens that holds and
the implied maximum concurrency. Those three numbers are the single best
summary of what a given model/GPU/config can serve.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import httpx


@dataclass(frozen=True)
class ServerConfig:
    model: str
    dtype: str = "float16"  # T4 (Turing) has no bfloat16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256  # set to 1 to effectively disable continuous batching
    port: int = 8000
    host: str = "0.0.0.0"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def build_command(cfg: ServerConfig) -> list[str]:
    cmd = [
        "vllm", "serve", cfg.model,
        "--dtype", cfg.dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--host", cfg.host,
        "--port", str(cfg.port),
    ]
    cmd.extend(cfg.extra_args)
    return cmd


@dataclass
class StartupInfo:
    attention_backend: str | None = None
    model_load_gib: float | None = None
    model_load_seconds: float | None = None
    kv_cache_gib: float | None = None
    kv_cache_tokens: int | None = None
    max_concurrency: float | None = None


_PATTERNS = {
    "attention_backend": (re.compile(r"Using (\S+) backend"), str),
    "model_load_gib": (re.compile(r"Model loading took ([\d.]+) GiB"), float),
    "model_load_seconds": (re.compile(r"Model loading took [\d.]+ GiB and ([\d.]+) seconds"), float),
    "kv_cache_gib": (re.compile(r"Available KV cache memory: ([\d.]+) GiB"), float),
    "kv_cache_tokens": (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda s: int(s.replace(",", ""))),
    "max_concurrency": (re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"), float),
}


def parse_startup_log(text: str) -> StartupInfo:
    info = StartupInfo()
    for name, (pattern, convert) in _PATTERNS.items():
        m = pattern.search(text)
        if m:
            setattr(info, name, convert(m.group(1)))
    return info


class VllmServer:
    """Runs `vllm serve` as a subprocess, tees its log to a file, waits for /health."""

    def __init__(self, cfg: ServerConfig, log_path: Path | str = "results/startup.log"):
        self.cfg = cfg
        self.log_path = Path(log_path)
        self.proc: subprocess.Popen | None = None

    def start(self, ready_timeout_s: float = 600) -> StartupInfo:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.log_path.open("w")
        self.proc = subprocess.Popen(build_command(self.cfg), stdout=log_file, stderr=subprocess.STDOUT)
        self._wait_ready(ready_timeout_s)
        return parse_startup_log(self.log_path.read_text())

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early; see {self.log_path}")
            try:
                if httpx.get(f"{self.cfg.base_url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM not ready after {timeout_s}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
