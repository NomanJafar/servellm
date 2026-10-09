# Concepts: what you need to understand before serving a model

This primer covers the ideas behind `servellm`. Read it before the code; the experiments in `notebooks/` are designed to make each of these visible in numbers.

## 1. A generation request is two very different workloads

An LLM request has two phases, and nearly every serving-system idea exists because they behave differently.

**Prefill** processes the whole prompt at once. All prompt tokens go through the model in one forward pass, producing the key/value tensors for every layer (the "KV cache") and the first output token. Prefill is *compute-bound*: thousands of tokens are multiplied through the weights in parallel, so the GPU's matrix units are the bottleneck. Its cost scales with prompt length. Time-to-first-token (TTFT) is mostly prefill time plus queueing.

**Decode** produces the remaining tokens one at a time. Each step feeds in a single token, attends over the cached K/V for everything before it, and emits one more token. For a single request the matmuls are tiny (one row times the weight matrix), so the GPU spends almost all its time *reading weights from memory*, not computing. Decode is *memory-bandwidth-bound*. Inter-token latency (ITL) is the per-step decode time.

The numbers for our setup make this concrete. Qwen2.5-1.5B-Instruct in FP16 is about 3.1 GB of weights. A T4 moves roughly 320 GB/s from its memory. Reading all weights once therefore takes at least 3.1 / 320 ≈ 10 ms. A single request decoding alone can never beat ~10 ms per token (~100 tokens/s) on this card, no matter how fast the arithmetic is, because every decode step has to stream the whole model through the chip to produce one token.

**The key insight:** that 10 ms read is the same whether it produces one token or sixty-four. If 64 requests are in flight and each decode step processes one token for each of them, the weights are read once and 64 tokens come out. Per-token cost drops almost 64×. This is why batching matters more for LLM serving than for almost any other workload: decode is so memory-bound that the arithmetic for extra sequences is nearly free until the batch gets large.

## 2. Static batching and why it wastes GPUs

The obvious way to batch is to collect N requests, run them together, and return when all are done. The problem: generations have different lengths. If one request wants 20 tokens and another wants 500, the first finishes at step 20 and its slot sits idle for 480 steps. New requests cannot join until the whole batch finishes. Measured utilisation under static batching with realistic length variation is often well under 50%.

## 3. Continuous batching

Continuous batching (also called iteration-level scheduling; introduced by Orca in 2022, implemented in vLLM, TGI, SGLang, TensorRT-LLM) makes the scheduling decision *every decode step* instead of every batch:

1. After each step, any sequence that emitted an end-of-sequence token or hit its max length leaves the batch immediately.
2. Waiting requests are admitted into the freed slots. A newly admitted request needs a prefill before it can decode; vLLM handles this by running prefill for new requests and decode for running ones in the same or adjacent steps (how exactly depends on the scheduler configuration, which project 8 explores).
3. Repeat.

The effect: the batch stays full as long as there is demand, and no request waits for an unrelated one to finish. Throughput (tokens/s across all users) rises steeply with concurrency while per-request ITL rises only gently, until a limit is reached.

What limits it? Two things:
- **Memory.** Every sequence in the batch holds a KV cache. More sequences means more cache. When memory runs out, new requests queue (or running ones get preempted).
- **Compute.** Eventually the batch is large enough that decode is no longer memory-bound; the matmuls become the bottleneck and ITL starts rising linearly with batch size.

On a T4 with a 1.5B model, memory is the constraint you will hit first. Which brings us to the KV cache.

## 4. The KV cache and why it dominates memory

For every token in every sequence, every layer stores a key vector and a value vector per KV head. The size per token is:

```
bytes_per_token = 2 (K and V) × n_layers × n_kv_heads × head_dim × bytes_per_element
```

Qwen2.5-1.5B uses grouped-query attention: 28 layers, 2 KV heads, head_dim 128, FP16 (2 bytes):

```
2 × 28 × 2 × 128 × 2 = 28,672 bytes ≈ 28 KB per token
```

That looks small. But with 64 concurrent sequences averaging 2,000 tokens of context each, that is 128,000 tokens × 28 KB ≈ 3.6 GB, already more than the model weights. Models without GQA (e.g. Llama-2-7B, 32 KV heads) spend 16× more per token. This is why project 3 is a calculator: KV cache, not weights, is what fills the GPU under load.

vLLM reserves a fixed fraction of GPU memory at startup (`--gpu-memory-utilization`, default 0.9), subtracts the weights and a profiled peak for activations, and carves the rest into KV cache blocks. It prints how many blocks it got and the implied maximum concurrency. Reading that log line is the first thing to do after launching a server.

## 5. PagedAttention in one paragraph

Before vLLM, servers allocated each request's KV cache as one contiguous region sized for its *maximum* possible length. Most requests never reached it, so most of the memory was reserved but empty; the vLLM paper measured 60–80% waste. PagedAttention borrows virtual memory from operating systems: the cache is split into fixed-size blocks (16 tokens each by default), a sequence's blocks need not be contiguous, and blocks are allocated only as tokens are actually generated. A per-sequence block table maps logical positions to physical blocks, and the attention kernel follows that table. Result: near-zero waste, so more sequences fit, so batching works harder. Blocks also enable sharing (identical prefixes can point at the same physical blocks; project 4) and preemption under pressure (project 9).

## 6. The OpenAI-compatible API

vLLM exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/v1/embeddings` with the same request and response shapes as OpenAI's API. This matters for infrastructure for a practical reason: every client library, gateway and evaluation tool already speaks this protocol, so a self-hosted model drops in behind the same client code as a hosted one. Project 13's gateway relies on this to swap providers transparently.

The parts that matter for measurement:
- `stream: true` returns server-sent events, one chunk per token (or small group). TTFT is the time to the first chunk with content; ITL is the gap between subsequent chunks. Without streaming you can only measure total latency.
- `usage` in the final response (or final chunk with `stream_options.include_usage`) reports `prompt_tokens` and `completion_tokens`, which you need for throughput and later for cost accounting (project 12).
- `max_tokens`, `temperature`, `seed`: fix them in benchmarks so runs are comparable. For latency measurement use `temperature: 0` or a fixed seed; the arithmetic cost does not depend on sampling but the *output length* does, and length variation swamps most other effects.

## 7. What the server tells you: `/metrics`

vLLM exports Prometheus metrics on `/metrics`. The ones this project watches:

| Metric | Meaning |
|---|---|
| `vllm:num_requests_running` | sequences currently in the batch |
| `vllm:num_requests_waiting` | queued, not yet admitted |
| `vllm:kv_cache_usage_perc` (or `gpu_cache_usage_perc` on older versions) | fraction of KV blocks in use |
| `vllm:time_to_first_token_seconds` | TTFT histogram, server-side |
| `vllm:time_per_output_token_seconds` | ITL histogram, server-side |
| `vllm:prompt_tokens_total`, `vllm:generation_tokens_total` | counters for throughput |

Server-side TTFT excludes network and client overhead; client-side TTFT includes them. Reporting both, and explaining the gap, is part of doing this honestly.

## 8. Hardware facts for the T4 that shape every decision

- 16 GB GDDR6, ~320 GB/s. Memory-bound decode tops out near 100 tokens/s for a single stream of a 1.5B FP16 model (section 1).
- Turing architecture (compute capability 7.5). vLLM supports it, but: **no bfloat16** (use `--dtype float16`), **no FlashAttention** (vLLM selects a fallback attention backend; expect a log line saying which), and **no FP8** (relevant to project 5).
- ~65 TFLOPS FP16 tensor-core peak, in practice far less. Prefill of a 2,000-token prompt through a 1.5B model is ~6 TFLOP of work, so TTFT for long prompts will be hundreds of milliseconds at best.
- Kaggle gives ~15 GB usable VRAM and a 12-hour session cap; Colab free gives a T4 with session limits that vary. Experiments must checkpoint results to disk as they go.

## 9. What the experiments in this project will show

1. **Launch and inspect.** Start vLLM with Qwen2.5-1.5B-Instruct, read the startup log: weights loaded, KV blocks allocated, max concurrency, attention backend chosen. Compare the block count against the hand calculation in section 4.
2. **Single-stream baseline.** One request at a time, streaming. Measure TTFT and ITL. Compare ITL against the ~10 ms bandwidth floor.
3. **Concurrency sweep.** 1, 2, 4, 8, 16, 32, 64 concurrent streams with fixed prompt and output lengths. Plot aggregate tokens/s and per-request ITL against concurrency. The throughput curve should rise steeply then flatten; ITL should stay nearly flat then climb. Where it bends is the point where decode stops being memory-bound or KV cache fills.
4. **Batching on vs off.** Run the sweep again with `--max-num-seqs 1` (effectively disabling batching). The gap between the two throughput curves is what continuous batching is worth on this hardware.
5. **Length variation.** Mix short and long generations. Confirm that short requests finish and leave without waiting for long ones (watch `num_requests_running` over time).

The full benchmark harness is project 2; this project uses a minimal client just sufficient to show these effects.

## Further reading

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023): the vLLM paper.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022): where continuous batching comes from.
- vLLM docs: *Engine Arguments* and *Metrics* pages for the current release.
