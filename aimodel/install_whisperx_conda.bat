@echo off
REM ========================================
REM WhisperX Conda Installation Script
REM Platform: Windows
REM ========================================

echo.
echo ========================================
echo WhisperX Conda Installation
echo ========================================
echo.

REM Activate conda environment
echo [1/7] Activating conda environment...
call conda activate whisperx_cuda
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    echo Please ensure 'whisperx_cuda' environment exists
    pause
    exit /b 1
)

REM Step 1: Remove conflicting packages
echo.
echo [2/7] Removing conflicting packages...
pip uninstall torch torchvision torchaudio whisperx faster-whisper numpy opencv-python -y
if errorlevel 1 echo WARNING: Some packages were not installed

REM Step 2: Install PyTorch via Conda
echo.
echo [3/7] Installing PyTorch 2.5.1 with CUDA 12.1...
call conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

REM Step 3: Install scientific libraries
echo.
echo [4/7] Installing scientific libraries...
call conda install numpy=1.26.4 scipy scikit-learn opencv ffmpeg -y
if errorlevel 1 (
    echo ERROR: Failed to install scientific libraries
    pause
    exit /b 1
)

REM Step 4: Install audio processing libraries
echo.
echo [5/7] Installing audio/video processing...
pip install librosa soundfile pyannote.audio --quiet
if errorlevel 1 echo WARNING: Some audio libraries had issues

REM Step 5: Install WhisperX and g2pk
echo.
echo [6/7] Installing WhisperX 3.7.4 and g2pk...
pip install --upgrade faster-whisper --quiet
pip install whisperx==3.7.4 --quiet
pip install g2pk --quiet
if errorlevel 1 (
    echo ERROR: Failed to install WhisperX stack
    pause
    exit /b 1
)

REM Step 6: Fix huggingface-hub compatibility
echo.
echo [7/7] Installing huggingface-hub compatibility fix...
pip install "huggingface-hub>=0.34.0,<1.0" --quiet

REM Verification
echo.
echo ========================================
echo Verification
echo ========================================
echo.

python -c "^
import torch; ^
print(f'PyTorch: {torch.__version__}'); ^
print(f'CUDA Available: {torch.cuda.is_available()}'); ^
if torch.cuda.is_available(): print(f'GPU: {torch.cuda.get_device_name(0)}')^
"

python -c "import whisperx; print('WhisperX: OK')" 2>nul || echo "WhisperX: FAILED"
python -c "from g2pk import G2p; print('g2pk: OK')" 2>nul || echo "g2pk: FAILED"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python -W ignore tests\test_whisperx_aligner.py
echo 2. Check: mobile_deepfake_detector\CONDA_INSTALLATION_SUCCESS.md
echo 3. Integrate into training pipeline
echo.

pause
