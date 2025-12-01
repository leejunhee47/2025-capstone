# WhisperX Conda Installation Guide

## Problem

Using `pip install` for WhisperX stack causes dependency version conflicts:
- `faster-whisper`: pip installs latest 1.1.0+, but WhisperX 3.1.1 requires 1.0.3
- `pytorch`: cu126 index doesn't have PyTorch 2.5.1, had to use cu121
- `numpy/opencv`: NumPy 2.x incompatible with OpenCV compiled for NumPy 1.x

**Solution:** Use Conda's superior dependency resolver instead of pip.

## Why Conda Works Better

1. **Unified Dependency Resolution:** Conda resolves ALL dependencies at once, not sequentially
2. **Version Constraint Propagation:** Conda respects pinned versions across entire stack
3. **Binary Compatibility:** Conda packages pre-compiled with compatible libs (libc, CUDA, etc.)
4. **No API Surprises:** Less chance of hidden API changes between patch versions

## Installation Steps

### Step 1: Remove Conflicting Packages

```bash
conda activate whisperx_cuda

# Uninstall pip-installed packages that caused conflicts
pip uninstall torch torchvision torchaudio whisperx faster-whisper -y

# Also remove problematic numpy/opencv if installed
pip uninstall numpy opencv-python -y
```

### Step 2: Install PyTorch via Conda (Recommended)

```bash
# Use conda instead of pip for PyTorch
# This ensures CUDA support and proper binary compatibility
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Why this works:**
- Conda's pytorch channel has pre-built wheels for ALL CUDA versions
- `pytorch-cuda=12.1` works with system CUDA 12.4/12.6 via forward compatibility
- Conda bundles compatible CUDA runtime libraries

### Step 3: Install Core Scientific Libraries

```bash
# Install NumPy and basic scientific stack via Conda
conda install numpy=1.26.4 scipy scikit-learn -y

# Install OpenCV via Conda (pre-compiled for NumPy 1.26.4)
conda install opencv -y
```

**Why this works:**
- Conda's opencv package is built against NumPy 1.26.4
- No manual version matching needed

### Step 4: Install Audio/Video Processing

```bash
# FFmpeg for audio extraction
conda install ffmpeg -y

# Install additional audio libraries
pip install librosa soundfile

# Pyannote for VAD (voice activity detection)
pip install pyannote.audio
```

### Step 5: Install WhisperX Stack (Carefully)

**Option A: From GitHub (Recommended)**
```bash
# Install faster-whisper 1.0.3 FIRST with version lock
pip install faster-whisper==1.0.3 --no-deps

# Then install WhisperX from GitHub with locked dependencies
pip install git+https://github.com/m-bain/whisperx.git@v3.1.1
```

**Option B: Using pip with version locks**
```bash
# Install exact versions with --no-deps to prevent auto-upgrade
pip install faster-whisper==1.0.3 --no-deps
pip install whisperx==3.1.1 --no-deps
```

**Why --no-deps works:**
- Forces pip to install ONLY the specified version
- Dependencies already satisfied by conda+pip earlier steps
- Prevents pip's auto-upgrade behavior

### Step 6: Install Korean Language Support

```bash
# g2pk for Korean pronunciation (Grapheme-to-Phoneme)
pip install g2pk

# Optional: Additional Korean NLP tools
pip install konlpy
```

### Step 7: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

# Test WhisperX import
python -c "import whisperx; print('WhisperX ✓')"

# Test g2pk
python -c "from g2pk import G2p; g2p = G2p(); print(g2p('안녕하세요'))"
```

## Complete Installation Script

Save as `install_whisperx_conda.sh` (Linux/Mac) or `.bat` (Windows):

**Windows (Batch):**
```batch
@echo off
echo ========================================
echo WhisperX Conda Installation
echo ========================================

echo Step 1: Remove conflicting packages...
call conda activate whisperx_cuda
pip uninstall torch torchvision torchaudio whisperx faster-whisper numpy opencv-python -y

echo Step 2: Install PyTorch via Conda...
call conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo Step 3: Install scientific libraries...
call conda install numpy=1.26.4 scipy scikit-learn opencv -y

echo Step 4: Install audio/video tools...
call conda install ffmpeg -y
call pip install librosa soundfile pyannote.audio

echo Step 5: Install WhisperX stack...
call pip install faster-whisper==1.0.3 --no-deps
call pip install git+https://github.com/m-bain/whisperx.git@v3.1.1

echo Step 6: Install Korean support...
call pip install g2pk konlpy

echo Step 7: Verify installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import whisperx; print('WhisperX ✓')"
python -c "from g2pk import G2p; print('g2pk ✓')"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
```

**Linux/Mac (Bash):**
```bash
#!/bin/bash
echo "========================================"
echo "WhisperX Conda Installation"
echo "========================================"

echo "Step 1: Remove conflicting packages..."
conda activate whisperx_cuda
pip uninstall torch torchvision torchaudio whisperx faster-whisper numpy opencv-python -y

echo "Step 2: Install PyTorch via Conda..."
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "Step 3: Install scientific libraries..."
conda install numpy=1.26.4 scipy scikit-learn opencv -y

echo "Step 4: Install audio/video tools..."
conda install ffmpeg -y
pip install librosa soundfile pyannote.audio

echo "Step 5: Install WhisperX stack..."
pip install faster-whisper==1.0.3 --no-deps
pip install git+https://github.com/m-bain/whisperx.git@v3.1.1

echo "Step 6: Install Korean support..."
pip install g2pk konlpy

echo "Step 7: Verify installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import whisperx; print('WhisperX ✓')"
python -c "from g2pk import G2p; print('g2pk ✓')"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
```

## Troubleshooting

### Issue: `torch.cuda.is_available()` still returns False

**Causes:**
1. Conda installed CPU-only PyTorch
2. CUDA library mismatch in system PATH

**Solutions:**
```bash
# Check what was installed
conda list | grep pytorch

# Reinstall with explicit CUDA:
conda uninstall pytorch -y
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'No CUDA')"
```

### Issue: `faster-whisper` keeps upgrading to 1.1.0+

**Root cause:** pip auto-upgrades dependencies unless --no-deps used

**Solution:**
```bash
# Force downgrade with version lock and no-deps
pip install faster-whisper==1.0.3 --force-reinstall --no-deps --no-build-isolation
```

### Issue: WhisperX import fails with `TranscriptionOptions` error

**Cause:** faster-whisper 1.1.0+ API differs from 1.0.3

**Solution:**
```bash
# Check installed version
pip show faster-whisper

# If 1.1.0+, remove and reinstall
pip uninstall faster-whisper -y
pip install faster-whisper==1.0.3 --no-deps
```

### Issue: OpenCV error about NumPy

**Cause:** NumPy version mismatch (2.x vs 1.x API)

**Solution:**
```bash
# Use conda's opencv (pre-compiled for NumPy 1.26.4)
pip uninstall opencv-python -y
conda install opencv -y
```

## Performance Verification

After installation, run the test suite:

```bash
cd E:\capstone\mobile_deepfake_detector
python tests\test_whisperx_aligner.py
```

Expected output:
```
======================================================================
WhisperX Phoneme Aligner Test Suite
======================================================================

GPU Availability Test
======================================================================
CUDA Available: True
GPU Device: NVIDIA GeForce RTX 3090
GPU Memory: 24.00 GB

g2pk Pronunciation Conversion Test
======================================================================
Testing pronunciation conversion:
  '안녕하세요' → 'ㄴ-ㅎ-ㅂ-ㅅ-ㅎ-ㅈ-ㅇ' (7 phonemes)
  '한국어' → 'ㅎ-ㄱ-ㄱ-ㅇ' (4 phonemes)
  ...

✓ g2pk working correctly
```

## Environment File (requirements.txt)

Save as `whisperx_conda_requirements.txt`:

```
# Core scientific stack (from conda)
# numpy==1.26.4
# scipy
# scikit-learn
# opencv-python (use conda install opencv instead)
# pytorch::pytorch==2.5.1
# pytorch::torchvision==0.20.1
# pytorch::torchaudio==2.5.1

# Audio/Video (pip)
librosa
soundfile
ffmpeg-python
pyannote.audio

# Speech processing (pip with version locks)
faster-whisper==1.0.3
whisperx==3.1.1  # or git+https://github.com/m-bain/whisperx.git@v3.1.1

# Korean language
g2pk
konlpy

# Development
jupyter
tensorboard
pytest
```

## Key Takeaways

| Issue | pip Solution | Conda Solution |
|-------|--------------|-----------------|
| Conflicting versions | Manual downgrade | Automatic resolution |
| NumPy API mismatch | Download specific OpenCV wheel | Pre-compiled binary |
| CUDA version mismatch | Use cu121 index | pytorch-cuda=12.1 option |
| Dependency cascades | --no-deps workaround | Built-in constraint solving |
| Version pinning | Fragile across updates | Guaranteed in lock files |

**Bottom Line:** Conda's superior dependency resolution eliminates the version hell that pip creates. Use Conda for the scientific stack (PyTorch, NumPy, SciPy, OpenCV), then pip for specialized packages (WhisperX, g2pk, pyannote).
