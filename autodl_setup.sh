#!/bin/bash
# AutoDL Setup Script for VACE with TFSA
# This script helps set up the environment on AutoDL platform

set -e  # Exit on error

echo "=========================================="
echo "VACE + TFSA AutoDL Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version || {
    echo "ERROR: Python not found!"
    echo "Please install Python 3.8+ first"
    exit 1
}
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "PyTorch installation completed!"
echo ""

# Install dependencies
echo "Installing VACE dependencies..."
pip install -r requirements.txt
echo ""

# Install additional packages for VACE
echo "Installing additional packages..."
pip install transformers
pip install accelerate
pip install safetensors
pip install huggingface-hub
echo ""

# Install huggingface-cli for model download
echo "Installing huggingface-cli..."
pip install -U huggingface_hub
echo ""

# Create models directory
echo "Creating models directory..."
mkdir -p models
echo ""

# Download LTX-Video-2B model
echo "=========================================="
echo "Model Download"
echo "=========================================="
echo ""
echo "Downloading LTX-Video-2B model from HuggingFace..."
echo "This will take several minutes depending on your connection speed."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create model directory
mkdir -p models/LTX-Video-2B

# Download using huggingface-cli
echo "Downloading model files..."
huggingface-cli download Lightricks/LTX-Video-2B \
    --local-dir models/LTX-Video-2B \
    --local-dir-use-symlinks False

echo ""
echo "Model download completed!"
echo ""

# Verify installation
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

echo "Checking TFSA module..."
python -c "
import sys
sys.path.insert(0, 'vace/models')
from tfsa_module import TrainingFreeSelfAttention, AdaptiveTFSA
print('✓ TFSA module imported successfully')
"

echo ""
echo "Checking VACE imports..."
python -c "
try:
    from vace.models.ltx.ltx_vace import LTXVace
    print('✓ VACE module can be imported')
except Exception as e:
    print(f'⚠ VACE import warning: {e}')
"

echo ""
echo "Running TFSA validation tests..."
python test_tfsa.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run VACE with TFSA:"
echo ""
echo "  python run_vace_with_tfsa.py \\"
echo "    --prompt \"A beautiful landscape\" \\"
echo "    --tfsa_enabled \\"
echo "    --tfsa_guidance 0.5 \\"
echo "    --output output.mp4"
echo ""
echo "For more information, see TFSA_README.md"
echo ""
