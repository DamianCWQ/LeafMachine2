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

## PlantSAM components
### 1. Install Dependancies
```
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 86827e2fbae8a293f61d51caa70a4b0602c04454
pip install --no-build-isolation -e .
```
### 2. Downloading Models
```
mkdir models
```
**SAM2 model weights** : [Download the model here](https://drive.google.com/file/d/1WN0pzBcQLIEF3TIMNj9JC7THtsnvds2i/view?usp=sharing)

**PlantSAM2** : [Download the model here](https://drive.google.com/file/d/1b57wlX9tCHRp4h92or41aRnBLA38rEfg/view?usp=sharing)

**YOLOv10** : [Download the model here](https://drive.google.com/file/d/1o-UcVMxktZQuz5DafjSR4T72gimdtujW/view?usp=sharing)


## Troubleshooting
**Manual Installation Issues:**
- Ensure Python 3.11 is installed (not 3.12+)
- On Windows, install Microsoft Visual C++ 14.0+ for pycocotools
- For CUDA issues, verify your GPU drivers match CUDA 12.1
