# LeafMachine2 Easy Installation Guide

Choose one of the following installation methods:

---

## Option 1: Docker Installation (Recommended)

Docker provides the easiest setup with all dependencies pre-configured.

### Prerequisites:
- Docker Desktop installed ([download here](https://www.docker.com/products/docker-desktop))
- For GPU support: NVIDIA GPU with drivers + NVIDIA Container Toolkit

### Using Docker Compose (Easiest)

**GPU Version:**
```bash
cd docker
docker compose -f compose.gpu.public.yml pull
docker compose -f compose.gpu.public.yml run lm2-gpu python test.py
```

**CPU Version:**
```bash
cd docker
docker compose -f compose.cpu.public.yml pull
docker compose -f compose.cpu.public.yml run lm2-cpu python test_cpu_only.py
```

### Using Docker CLI

**GPU Version:**
```bash
# Pull the image
docker pull ghcr.io/gene-weaver/leafmachine2:gpu-latest

# Run with GPU support
docker run --gpus all -v .:/app -w /app ghcr.io/gene-weaver/leafmachine2:gpu-latest python3 test.py
```

**CPU Version:**
```bash
# Pull the image
docker pull ghcr.io/gene-weaver/leafmachine2:cpu-latest

# Run CPU version
docker run -v .:/app -w /app ghcr.io/gene-weaver/leafmachine2:cpu-latest python3 test_cpu_only.py
```

**Notes:**
- Replace `test.py` or `test_cpu_only.py` with your actual script name
- The `-v .:/app` mounts your current directory into the container
- For Windows, use PowerShell or add `${PWD}` instead of `.` for the volume mount

---

## Option 2: Manual Python Installation

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

---

## Troubleshooting

**Docker Issues:**
- Ensure Docker Desktop is running
- For GPU: Verify NVIDIA Container Toolkit is installed with `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

**Manual Installation Issues:**
- Ensure Python 3.11 is installed (not 3.12+)
- On Windows, install Microsoft Visual C++ 14.0+ for pycocotools
- For CUDA issues, verify your GPU drivers match CUDA 12.1