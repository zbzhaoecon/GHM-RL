# GPU Setup Guide for Google Colab

If you're experiencing issues where training runs on CPU instead of GPU (even with `--device cuda`), follow these steps:

## Step 1: Verify GPU Runtime

1. In Google Colab, go to **Runtime** > **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (preferably T4 or better)
3. Click **Save**
4. If you changed the runtime, click **Runtime** > **Restart runtime**

## Step 2: Check CUDA Availability

Run this in a Colab cell:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("\n✓ GPU is ready!")
else:
    print("\n✗ CUDA not available - PyTorch will use CPU")
    print("Possible fixes:")
    print("1. Ensure GPU runtime is selected (Runtime > Change runtime type)")
    print("2. Reinstall PyTorch with CUDA support (see Step 3)")
```

## Step 3: Reinstall PyTorch with CUDA (if needed)

If Step 2 shows `CUDA available: False`, run:

```python
# Uninstall current PyTorch
!pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 11.8 support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

**Note**: After reinstalling, you may need to restart the runtime:
- **Runtime** > **Restart runtime**
- Re-run your imports

## Step 4: Test GPU Training

Run this quick test to ensure GPU is being used:

```python
import torch

# Create model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensors on GPU
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Perform operation (this should use GPU)
z = torch.matmul(x, y)

print(f"Result tensor device: {z.device}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

if z.device.type == 'cuda':
    print("\n✓ GPU is working correctly!")
else:
    print("\n✗ Still using CPU")
```

## Step 5: Verify Training Uses GPU

When running hyperparameter search or training:

```python
# Run with GPU
!python scripts/hyperparameter_search.py \
    --search_type random \
    --n_trials 2 \
    --n_iterations 100 \
    --device cuda

# Check GPU usage in another cell while training:
!nvidia-smi
```

You should see:
- GPU memory usage increase in `nvidia-smi`
- "Using device: cuda" in training output
- GPU RAM usage in Colab's resource panel (not System RAM)

## Common Issues

### Issue: "CUDA available: False" even with GPU runtime

**Solution**:
1. Restart runtime: **Runtime** > **Restart runtime**
2. Reinstall PyTorch (Step 3)
3. If still not working, try a different GPU: **Runtime** > **Change runtime type** > Try different GPU types

### Issue: Training shows "Using device: cuda" but uses system RAM

**Solution**:
1. This shouldn't happen with the current codebase (all components properly handle device)
2. Check training logs for errors
3. Verify with `nvidia-smi` that GPU is actually being used

### Issue: "RuntimeError: CUDA out of memory"

**Solution**:
1. Reduce `--n_trajectories` (e.g., 250 instead of 500)
2. Reduce network size: `--policy_hidden 32 32 --value_hidden 32 32`
3. With parallel execution: reduce `--n_workers` (e.g., 2 instead of 4)

### Issue: GPU memory not released between runs

**Solution**:
```python
import torch
torch.cuda.empty_cache()
```

## Quick Diagnostic Script

Run this comprehensive diagnostic:

```python
!python scripts/check_cuda.py
```

This will check:
- PyTorch version
- CUDA availability
- GPU information
- GPU operations
- Memory usage

## Using the Setup Script

For automatic setup, run:

```bash
!bash scripts/setup_colab_gpu.sh
```

This will:
- Check CUDA installation
- Detect GPU
- Verify PyTorch CUDA support
- Reinstall PyTorch if needed

---

## Expected Resource Usage with GPU

When training properly uses GPU, you should see:

**Google Colab Resources Panel:**
- **System RAM**: 2-4 GB (low usage)
- **GPU RAM**: 5-12 GB (high usage, depending on hyperparameters)
- **Disk**: Normal usage

**nvidia-smi output:**
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1234      C   python3                          8192MiB |
+-----------------------------------------------------------------------------+
```

If you see high **System RAM** usage instead, PyTorch is using CPU despite `--device cuda`.
