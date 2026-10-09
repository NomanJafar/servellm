from servellm.launch import ServerConfig, build_command, parse_startup_log

# Trimmed from a real vLLM V1 startup log. The notebook saves the live log to
# results/startup.log so this fixture can be refreshed when vLLM changes its wording.
SAMPLE_LOG = """\
INFO 10-10 12:00:01 [cuda.py:180] Using FLEX_ATTENTION backend.
INFO 10-10 12:00:09 [gpu_model_runner.py:1200] Model loading took 2.8882 GiB and 6.412 seconds
INFO 10-10 12:00:15 [gpu_worker.py:260] Available KV cache memory: 9.58 GiB
INFO 10-10 12:00:15 [kv_cache_utils.py:900] GPU KV cache size: 350,816 tokens
INFO 10-10 12:00:15 [kv_cache_utils.py:905] Maximum concurrency for 4,096 tokens per request: 85.65x
INFO 10-10 12:00:20 [api_server.py:1500] Starting vLLM API server on http://0.0.0.0:8000
"""


def test_build_command_uses_vllm_serve_with_fp16_defaults():
    cfg = ServerConfig(model="Qwen/Qwen2.5-1.5B-Instruct")
    cmd = build_command(cfg)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"]
    assert "--dtype" in cmd and cmd[cmd.index("--dtype") + 1] == "float16"
    assert cmd[cmd.index("--port") + 1] == "8000"


def test_build_command_passes_batching_and_memory_knobs():
    cfg = ServerConfig(
        model="m", max_num_seqs=1, max_model_len=2048, gpu_memory_utilization=0.85, port=9000
    )
    cmd = build_command(cfg)
    assert cmd[cmd.index("--max-num-seqs") + 1] == "1"
    assert cmd[cmd.index("--max-model-len") + 1] == "2048"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.85"
    assert cmd[cmd.index("--port") + 1] == "9000"


def test_build_command_appends_extra_args_last():
    cfg = ServerConfig(model="m", extra_args=("--enable-prefix-caching",))
    assert build_command(cfg)[-1] == "--enable-prefix-caching"


def test_parse_startup_log_extracts_kv_cache_and_backend():
    info = parse_startup_log(SAMPLE_LOG)
    assert info.attention_backend == "FLEX_ATTENTION"
    assert info.kv_cache_tokens == 350_816
    assert info.kv_cache_gib == 9.58
    assert info.max_concurrency == 85.65
    assert info.model_load_gib == 2.8882
    assert info.model_load_seconds == 6.412


def test_parse_startup_log_tolerates_missing_lines():
    info = parse_startup_log("INFO something unrelated\n")
    assert info.attention_backend is None
    assert info.kv_cache_tokens is None
