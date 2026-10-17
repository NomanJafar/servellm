from servellm.launch import ServerConfig, build_command, parse_startup_log

# Trimmed from a real vLLM V1 startup log. The notebook saves the live log to
# results/startup.log so this fixture can be refreshed when vLLM changes its wording.
SAMPLE_LOG = """\
INFO 10-10 12:00:01 [cuda.py:526] Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION'].
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
    assert info.attention_backend == "TRITON_ATTN"
    assert info.kv_cache_tokens == 350_816
    assert info.kv_cache_gib == 9.58
    assert info.max_concurrency == 85.65
    assert info.model_load_gib == 2.8882
    assert info.model_load_seconds == 6.412


def test_parse_startup_log_tolerates_missing_lines():
    info = parse_startup_log("INFO something unrelated\n")
    assert info.attention_backend is None
    assert info.kv_cache_tokens is None


def test_parse_startup_log_accepts_older_backend_wording():
    assert parse_startup_log("INFO [cuda.py:180] Using FLEX_ATTENTION backend.\n").attention_backend == "FLEX_ATTENTION"


def test_server_env_is_merged_into_the_process_environment(monkeypatch, tmp_path):
    # Two replicas on two GPUs need different CUDA_VISIBLE_DEVICES; the env must be captured
    # per server at spawn time, not read from os.environ when start() happens to run.
    import subprocess

    from servellm.launch import VllmServer

    seen = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(cmd, stdout=None, stderr=None, env=None, **kw):
        seen["env"] = env
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(VllmServer, "_wait_ready", lambda self, t: None)
    s = VllmServer(ServerConfig(model="m"), log_path=tmp_path / "s.log", env={"CUDA_VISIBLE_DEVICES": "1"})
    s.start()
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "1" and "PATH" in seen["env"]


def test_kill_takes_the_whole_process_group_down(monkeypatch, tmp_path):
    """vLLM's API server forks an EngineCore child that owns the GPU memory. SIGKILL on the parent alone
    leaves that child alive (inferchaos run 3: the restarted server found 1.6 of 14.6 GiB free)."""
    import os
    import signal
    import time

    from servellm import launch
    from servellm.launch import VllmServer

    # a parent that forks a child, then idles: the shape of `vllm serve`
    monkeypatch.setattr(launch, "build_command", lambda cfg: ["bash", "-c", "sleep 300 & echo $! > child.pid; wait"])
    monkeypatch.setattr(VllmServer, "_wait_ready", lambda self, t: None)
    monkeypatch.chdir(tmp_path)
    s = VllmServer(ServerConfig(model="m"), log_path=tmp_path / "s.log")
    s.start()
    for _ in range(100):
        if (tmp_path / "child.pid").exists() and (tmp_path / "child.pid").read_text().strip():
            break
        time.sleep(0.02)
    child = int((tmp_path / "child.pid").read_text())
    os.kill(child, 0)                       # alive
    s.kill()
    assert s.proc.poll() is not None
    time.sleep(0.2)
    try:
        os.kill(child, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    if alive:                               # a zombie still answers signal 0; make sure it is at least not running
        from pathlib import Path

        stat = Path(f"/proc/{child}/stat")
        state = stat.read_text().split()[2] if stat.exists() else "?"
        assert state in ("Z", "?"), f"child {child} still running (state {state})"
    assert signal.SIGKILL  # keep the import honest
