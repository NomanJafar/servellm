Run 4 (2026-10-11): torchaudio now 2.11.0+cu129 (fixed), but torchcodec was re-pulled from PyPI as 0.17.0 by vLLM's
`torchcodec>=0.14` requirement during the vLLM install; the PyPI wheel links CUDA 13 (`libnvrtc.so.13`), same OSError
as run 3. The later cu129-index install was a no-op ("already satisfied"). The `!python -c` import check printed the
failure but, being a shell line, did not stop the cell. Fix for run 5: uninstall torchcodec after the vLLM install;
check with a real Python import.
