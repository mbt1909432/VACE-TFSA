@echo off
REM VACE + TFSA AutoDL Setup Script for Windows (local testing)
REM This script helps set up the environment for local testing

echo ==========================================
echo VACE + TFSA Setup Script (Windows)
echo ==========================================
echo.

REM Check Python
echo Checking Python version...
python --version || (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ first
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch
echo Installing PyTorch...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo PyTorch installation completed!
echo.

REM Install dependencies
echo Installing VACE dependencies...
python -m pip install -r requirements.txt
echo.

REM Install additional packages
echo Installing additional packages...
python -m pip install transformers accelerate safetensors huggingface-hub
echo.

REM Install huggingface-cli
echo Installing huggingface-cli...
python -m pip install -U huggingface_hub
echo.

REM Create models directory
if not exist models mkdir models
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To use TFSA, you can run the test:
echo   python test_tfsa.py
echo.
echo To download models, use huggingface-cli:
echo   huggingface-cli download Lightricks/LTX-Video-Video-2B --local-dir models/LTX-Video-2B
echo.
echo For AutoDL deployment, use autodl_setup.sh
echo.
pause
