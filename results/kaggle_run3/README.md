Run 3 (2026-10-11): cu129 wheel installed and imported fine (torch 2.13.0+cu129, CUDA available, torchvision 0.28.0+cu129),
driver 580.178.04 (CUDA 13.0 capable, so CUDA 13 itself was never the problem). `vllm serve` then died importing the
image's preinstalled torchcodec: `OSError: libnvrtc.so.13: cannot open shared object file` (torchcodec built for CUDA 13,
torch now cu129). vLLM guards that import for ImportError/RuntimeError but not OSError. Fix for run 4: uninstall the
image's torchaudio and torchcodec before installing so pip resolves matching ones from the cu129 index.
