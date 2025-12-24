"""
Quick diagnostic script to check CUDA availability and usage.

Run this before running hyperparameter search to verify GPU setup.
"""

import torch
import sys

print("=" * 80)
print("CUDA Diagnostics")
print("=" * 80)

# Check PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    # Test tensor creation and operations
    print("\n" + "-" * 80)
    print("Testing GPU operations...")
    print("-" * 80)

    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        # Perform operation
        z = torch.matmul(x, y)

        print("✓ Successfully created tensors on GPU")
        print("✓ Successfully performed matrix multiplication on GPU")
        print(f"✓ Result tensor device: {z.device}")

        # Check memory usage after operation
        print(f"\nMemory allocated after operation: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    except Exception as e:
        print(f"✗ GPU operation failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ CUDA is working correctly!")
    print("=" * 80)
    print("\nYou can use --device cuda for training.")

else:
    print("\n" + "=" * 80)
    print("✗ CUDA is NOT available")
    print("=" * 80)
    print("\nPossible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("   Fix: Reinstall PyTorch with CUDA:")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("\n2. No GPU available in this environment")
    print("   Fix: Use a GPU runtime (in Colab: Runtime > Change runtime type > GPU)")
    print("\n3. CUDA drivers not installed")
    print("   Fix: Install NVIDIA CUDA drivers")
    print("\nFor now, use --device cpu for training.")
    sys.exit(1)
