Run 1 (2026-10-11): FAILED at cell 3 with `ModuleNotFoundError: No module named 'servellm.launch'`.
Cause: `pip install -e` inside the running kernel adds a `.pth` path entry that is only read at interpreter start,
so the package was invisible; the cloned repo directory then matched as an empty namespace package, hence the
error naming `servellm.launch` rather than `servellm`. Fix: non-editable install, clone under a different name.
vLLM 0.31.0 + torch 2.13.0+cu130 installed fine on the Kaggle image.
