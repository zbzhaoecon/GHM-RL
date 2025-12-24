#!/bin/bash
#
# Setup script for Google Colab to ensure GPU/CUDA is properly configured
#
# Run this in a Colab cell with: !bash scripts/setup_colab_gpu.sh
#

echo "=========================================="
echo "Google Colab GPU Setup"
echo "=========================================="

# Check if running on Colab
if [ -d "/content" ]; then
    echo "✓ Running on Google Colab"
else
    echo "⚠ Not running on Google Colab"
fi

# Check CUDA installation
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version | head -n 4
    echo "✓ CUDA compiler found"
else
    echo "✗ CUDA compiler not found"
fi

# Check GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "✗ No GPU detected"
    echo ""
    echo "Fix: In Colab, go to Runtime > Change runtime type > Hardware accelerator > GPU"
    exit 1
fi

# Check PyTorch
echo ""
echo "Checking PyTorch..."
python3 << EOF
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch with CUDA support")
    sys.exit(0)
else:
    print("✗ PyTorch does NOT have CUDA support")
    print("")
    print("This usually means PyTorch was installed without CUDA.")
    print("Reinstalling PyTorch with CUDA support...")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Reinstalling PyTorch with CUDA support..."
    pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    echo ""
    echo "Verifying installation..."
    python3 << EOF
import torch
import sys

if torch.cuda.is_available():
    print("✓ PyTorch reinstalled successfully with CUDA support")
    print(f"CUDA version: {torch.version.cuda}")
    sys.exit(0)
else:
    print("✗ Still no CUDA support after reinstall")
    print("You may need to restart the runtime:")
    print("Runtime > Restart runtime")
    sys.exit(1)
EOF
fi

echo ""
echo "=========================================="
echo "✓ GPU setup complete!"
echo "=========================================="
echo ""
echo "You can now run training with --device cuda"
