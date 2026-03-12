# LeafMachine2 Easy Installation Guide

## Manual Python Installation

For users who prefer a local Python environment.

### 1. Prerequisites:
- Python 3.11 installed
- NVIDIA GPU with drivers installed (for GPU support)
- CUDA 12.1 toolkit (optional but recommended for best performance)

### 2. Create and activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Upgrade pip and install uv (fast package installer)
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install uv
```

### 4. Install main dependencies
```powershell
uv pip install -r requirements.txt
```
*Note: Use `pip install` if you prefer not to install uv*

### 5. Install specific required packages
```powershell
uv pip install "git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools"
uv pip install "pycocotools>=2.0.5"
uv pip install "opencv-contrib-python-headless==4.7.0.72"
uv pip install "vit-pytorch==0.37.1"
```

### 6. Install PyTorch with CUDA 12.1 support
```powershell
uv pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 7. Run the microservice:
`uvicorn api.main:app --reload --port 9000` to run the microservice

---

## Troubleshooting
**Manual Installation Issues:**
- Ensure Python 3.11 is installed (not 3.12+)
- On Windows, install Microsoft Visual C++ 14.0+ for pycocotools
- For CUDA issues, verify your GPU drivers match CUDA 12.1
