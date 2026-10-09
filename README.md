# servellm

Self-hosted OpenAI-compatible LLM serving on vLLM with continuous batching: setup, config walkthrough and measured batching behaviour on a single T4.

Project 1 of 15 in [inference-infra-portfolio](https://github.com/NomanJafar/inference-infra-portfolio).

## What this builds

Serve an open model with vLLM behind an OpenAI-compatible API with continuous batching enabled, and measure what batching actually does to latency and throughput.

## Why it matters

You cannot optimize what you have never served.

## Status

Code and tests complete; waiting on the first GPU run. `CONCEPTS.md` is written. Results land in `results/` and in the section below once the Kaggle notebook has been run.

## What is in here

| Module | Purpose |
|---|---|
| `servellm.launch` | `ServerConfig` → `vllm serve` command; `VllmServer` starts the process, waits for `/health`, and parses the startup log for KV cache size, token capacity, max concurrency and attention backend |
| `servellm.client` | Minimal streaming `/v1/chat/completions` client that timestamps every chunk and derives TTFT, per-token ITL and token counts |
| `servellm.metrics` | Scrapes `/metrics` and maps the handful of series we care about (running, waiting, KV usage, token counters, server-side TTFT/ITL) to stable names across vLLM versions |
| `servellm.sweep` | Closed-loop load generator: concurrency sweep with a `/metrics` timeline, and a mixed-length run; CLI via `python -m servellm.sweep` |
| `servellm.plots` | Throughput / ITL / TTFT vs concurrency, and the mixed-length finish-time and batch-occupancy charts |

## Experiments

1. **Launch and inspect**: read the KV cache numbers vLLM prints and check them against the hand calculation (≈28 KB per token for Qwen2.5-1.5B in FP16).
2. **Single-stream baseline**: TTFT and ITL for one request at a time; compare ITL to the ~10 ms weight-read floor on a T4.
3. **Concurrency sweep** at 1…64 concurrent streams with fixed prompt and output lengths.
4. **Batching off** (`--max-num-seqs 1`): the same sweep; the gap between the curves is what continuous batching is worth.
5. **Mixed lengths**: short and long requests launched together; short ones should finish early and free their slots.

## Layout

```
CONCEPTS.md     the theory behind the project, written before the code
src/servellm/   implementation
tests/          CPU-only tests, runnable on a laptop
notebooks/      GPU notebooks (Kaggle / Colab) that run the experiments and write results/
results/        measured numbers and plots
```

## Running

Tests (no GPU, no vLLM needed; the client is exercised against a fake SSE server):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Against a live server (any OpenAI-compatible endpoint works for the client; the startup-log parsing is vLLM-specific):

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --max-model-len 4096
python -m servellm.sweep --out results/sweep.json sweep --levels 1,2,4,8,16,32,64
python -m servellm.sweep --out results/mixed.json mixed --short 16 --long 512 --n-each 8
```

Full run on a free GPU: open `notebooks/kaggle_servellm.ipynb` on Kaggle (T4) or Colab. It installs vLLM 0.31.0, runs all five experiments and zips `results/`.

## Hardware notes

Developed against free-tier GPUs (Kaggle and Colab T4, 16 GB). Where a T4 limits what can be measured (no FP8, Turing-era kernels), the README says so rather than papering over it.

## License

MIT
