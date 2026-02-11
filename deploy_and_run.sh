#!/bin/bash
# Complete AutoDL Deployment Script for VACE + TFSA
# This script handles the full deployment workflow:
# 1. Enable network acceleration
# 2. Pull latest code
# 3. Download LTX-Video model
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
# Step 4: Download LTX-Video Model (2B version only)
# ============================================
echo "[4/5] Checking LTX-Video model..."

if [ -f "models/LTX-Video/ltx-video-2b-v0.9.safetensors" ]; then
    echo "✓ Model already exists, skipping download"
else
    echo "Downloading LTX-Video 2B model from HuggingFace..."
    echo "Repository: Lightricks/LTX-Video"
    echo "Model: ltx-video-2b-v0.9.safetensors (minimal version, ~5GB)"
    echo "This will take several minutes depending on your connection speed..."
    echo ""

    # Create model directory
    mkdir -p models/LTX-Video

    # Try with token if provided
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "Using HuggingFace token from environment..."
        python << 'EOF'
from huggingface_hub import hf_hub_download
import os, sys

try:
    os.makedirs('models/LTX-Video', exist_ok=True)
    print("Downloading ltx-video-2b-v0.9.safetensors...")
    hf_hub_download(
        repo_id='Lightricks/LTX-Video',
        filename='ltx-video-2b-v0.9.safetensors',
        local_dir='models/LTX-Video',
        token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    print("\n✓ Model download complete!")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    sys.exit(1)
EOF
    else
        # Try without token
        python << 'EOF'
from huggingface_hub import hf_hub_download
import os, sys

try:
    os.makedirs('models/LTX-Video', exist_ok=True)
    print("Downloading ltx-video-2b-v0.9.safetensors...")
    hf_hub_download(
        repo_id='Lightricks/LTX-Video',
        filename='ltx-video-2b-v0.9.safetensors',
        local_dir='models/LTX-Video'
    )
    print("\n✓ Model download complete!")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nIf download fails, you can:")
    print("1. Generate a token at https://huggingface.co/settings/tokens")
    print("2. Run: export HUGGINGFACE_TOKEN=your_token_here")
    print("3. Run this script again")
    print("\nOr download manually from:")
    print("  https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors")
    sys.exit(1)
EOF
    fi

    if [ $? -eq 0 ]; then
        echo "✓ Model downloaded successfully"
        ls -lh models/LTX-Video/ltx-video-2b-v0.9.safetensors
    else
        echo "✗ Model download failed"
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
