# Environment

The supported installation path from the repository root is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

`requirements.txt` installs the training/runtime extra, while
`requirements-dev.txt` installs both the training and test extras. Each file
delegates to `pyproject.toml`, which remains the single authoritative dependency
definition.

Alternatively:

```powershell
conda env create -f environment.yml
conda activate em-connectome
```

The migration and validation machine used Python 3.11.9, PyTorch
2.11.0+cu128, CUDA 12.8, cuDNN 9.19, and an NVIDIA GeForce RTX 5080 Laptop
GPU. The complete curated version list is in
`requirements/reference-environment.txt`.

PyTorch builds are platform-specific. Install a CUDA build compatible with the
target driver before `pip install -e ".[train]"` when the default resolver does
not select the intended accelerator build. CPU execution is supported for
verification and small smoke runs, but the formal 3×3×5 regression is intended
for CUDA.

Every training run writes `environment.json`, including Python, package, CUDA,
cuDNN, device, and deterministic-mode information. This records the actual
runtime rather than assuming it matches the reference file.
