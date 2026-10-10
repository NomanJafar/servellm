Run 2 (2026-10-11): install fix from run 1 worked (`servellm.launch` imported), but `vllm serve` exited at import:
`RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions. PyTorch has CUDA
version 13.0 whereas TorchAudio has CUDA version 12.8.` The default `vllm==0.31.0` PyPI wheel pulled torch 2.13+cu130
onto a Kaggle image whose preinstalled torchaudio is cu128; transformers imports torchaudio when present.
GPU was a Tesla T4 (15360 MiB, compute 7.5). Fix for run 3: cu129 release wheel + cu129 PyTorch index, uninstall torchaudio.
