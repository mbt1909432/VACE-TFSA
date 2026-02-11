#!/bin/bash
# Complete AutoDL Deployment Script for VACE + TFSA
# This script handles the full deployment workflow:
# 1. Enable network acceleration
# 2. Pull latest code
# 3. Download LTX-Video-2B model
# 4. Run TFSA validation tests
# 5. Execute video generation with TFSA

set -e  # Exit on error

echo "=========================================="
echo "VACE + TFSA AutoDL Deployment"
echo "=========================================="
echo ""

# ============================================
# Step 1: Enable Network Acceleration
# ============================================
echo "[1/5] Enabling academic network acceleration..."
source /etc/network_turbo
echo "✓ Network acceleration enabled"
echo ""

# ============================================
# Step 2: Pull Latest Code
# ============================================
echo "[2/5] Pulling latest code from GitHub..."
git pull origin main
echo "✓ Code updated"
echo ""

# ============================================
# Step 3: Run TFSA Validation Tests
# ============================================
echo "[3/5] Running TFSA validation tests..."
python test_tfsa.py
echo "✓ TFSA validation passed"
echo ""

# ============================================
# Step 4: Download LTX-Video-2B Model
# ============================================
echo "[4/5] Checking LTX-Video-2B model..."

if [ -d "models/LTX-Video-2B" ] && [ "$(ls -A models/LTX-Video-2B)" ]; then
    echo "✓ Model already exists, skipping download"
else
    echo "Attempting to download LTX-Video-2B model from HuggingFace..."
    echo "Note: LTX-Video-2B may require accepting the model license on HuggingFace."
    echo "      Visit: https://huggingface.co/Lightricks/LTX-Video-2B"
    echo ""
    echo "This will take several minutes (model size: ~5GB)..."

    # Try with token if provided
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "Using HuggingFace token from environment..."
        python << 'EOF'
from huggingface_hub import snapshot_download
import os, sys

try:
    print("Starting download with token...")
    snapshot_download(
        repo_id='Lightricks/LTX-Video-2B',
        local_dir='models/LTX-Video-2B',
        local_dir_use_symlinks=False,
        token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    print("\n✓ Model download complete!")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nIf you see '401 Unauthorized', you need to:")
    print("1. Visit https://huggingface.co/Lightricks/LTX-Video-2B")
    print("2. Accept the model license")
    print("3. Generate a token at https://huggingface.co/settings/tokens")
    print("4. Run: export HUGGINGFACE_TOKEN=your_token_here")
    print("5. Run this script again")
    sys.exit(1)
EOF
    else
        # Try without token
        python << 'EOF'
from huggingface_hub import snapshot_download
import sys

try:
    print("Starting download (without token)...")
    snapshot_download(
        repo_id='Lightricks/LTX-Video-2B',
        local_dir='models/LTX-Video-2B',
        local_dir_use_symlinks=False
    )
    print("\n✓ Model download complete!")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nLTX-Video-2B requires accepting the model license:")
    print("1. Visit https://huggingface.co/Lightricks/LTX-Video-2B")
    print("2. Click 'Agree and access repository'")
    print("3. Generate a token at https://huggingface.co/settings/tokens")
    print("4. Run: export HUGGINGFACE_TOKEN=your_token_here")
    print("5. Run this script again")
    print("\nAlternatively, download the model manually and place it in models/LTX-Video-2B/")
    sys.exit(1)
EOF
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Model downloaded successfully"
    else
        echo ""
        echo "=========================================="
        echo "Model Download Failed - Manual Setup Required"
        echo "=========================================="
        echo ""
        echo "Option 1: Use HuggingFace Token"
        echo "  export HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        echo "  bash deploy_and_run.sh"
        echo ""
        echo "Option 2: Download Manually"
        echo "  1. Visit: https://huggingface.co/Lightricks/LTX-Video-2B"
        echo "  2. Accept license and download model files"
        echo "  3. Upload to: models/LTX-Video-2B/"
        echo ""
        exit 1
    fi
fi
echo ""

# ============================================
# Step 5: Run Video Generation with TFSA
# ============================================
echo "[5/5] Running video generation with TFSA..."
echo ""

# Default parameters (can be overridden by arguments)
PROMPT="A beautiful landscape with mountains and a lake, cinematic lighting"
TFSA_ENABLED=true
TFSA_GUIDANCE=0.5
TFSA_MODE=augment
OUTPUT="output_tfsa.mp4"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --tfsa_guidance)
            TFSA_GUIDANCE="$2"
            shift 2
            ;;
        --tfsa_mode)
            TFSA_MODE="$2"
            shift 2
            ;;
        --no_tfsa)
            TFSA_ENABLED=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--prompt PROMPT] [--output FILE] [--tfsa_guidance VALUE] [--tfsa_mode MODE] [--no_tfsa]"
            exit 1
            ;;
    esac
done

echo "Generation parameters:"
echo "  Prompt: $PROMPT"
echo "  TFSA Enabled: $TFSA_ENABLED"
if [ "$TFSA_ENABLED" = true ]; then
    echo "  TFSA Guidance: $TFSA_GUIDANCE"
    echo "  TFSA Mode: $TFSA_MODE"
fi
echo "  Output: $OUTPUT"
echo ""

# Build command
CMD="python run_vace_with_tfsa.py --prompt \"$PROMPT\" --output $OUTPUT"

if [ "$TFSA_ENABLED" = true ]; then
    CMD="$CMD --tfsa_enabled --tfsa_guidance $TFSA_GUIDANCE --tfsa_mode $TFSA_MODE"
fi

echo "Executing..."
echo $CMD
echo ""

# Run inference
eval $CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Video generation completed!"
    echo "=========================================="
    echo "Output file: $OUTPUT"
    echo ""
    echo "File size:"
    ls -lh "$OUTPUT" | awk '{print "  " $5}'
    echo ""
    echo "To generate more videos, run:"
    echo "  ./deploy_and_run.sh --prompt \"Your prompt\" --output output.mp4"
else
    echo ""
    echo "✗ Video generation failed"
    exit 1
fi
