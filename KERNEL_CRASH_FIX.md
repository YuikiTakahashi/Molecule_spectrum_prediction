# Kernel Crash Fix - Summary

## Problem
The Jupyter kernel was dying when trying to run the `initialize_state()` cell. This typically indicates:
- Segmentation fault (C extension crash)
- Memory exhaustion
- Infinite recursion
- Stack overflow

## Root Causes Identified

1. **PyTorch Import Issue**: `Energy_Levels.py` was importing torch unconditionally, which could cause a segfault if PyTorch wasn't properly installed
2. **Unsafe Tensor Conversion**: The `diagonalize()` function was calling `.numpy()` directly on tensors without `.detach().cpu()` first
3. **No Fallback Mechanism**: If torch failed, there was no graceful fallback to numpy/scipy
4. **Memory Issues**: Large matrices during initialization could cause memory problems

## Fixes Applied

### 1. Made PyTorch Import Optional in `Energy_Levels.py`
- Changed from `import torch` to a try/except block
- Added `TORCH_AVAILABLE` flag to check if torch is usable
- This prevents kernel crashes if PyTorch isn't installed or is corrupted

### 2. Fixed Tensor Conversion
- Changed `.numpy()` to `.detach().cpu().numpy()` in both `diagonalize()` and `diagonalize_batch()`
- This ensures safe conversion even if tensors require gradients or are on GPU

### 3. Added Fallback Mechanisms
- `diagonalize()` functions now fall back to numpy if torch isn't available
- Initialization uses `numpy` method by default (more stable)
- Added error handling with fallback to scipy, then torch

### 4. Made Initialization Safer
- Changed default method in `__init__` from `torch` to `numpy`
- Added matrix size checking before diagonalization
- Added memory error handling with helpful error messages
- Added try/except blocks with fallback chain: numpy → scipy → torch

### 5. Updated Notebook
- Made torch import optional in the first cell
- Added graceful error handling in diagonalization functions
- Added diagnostic messages about which method is being used

## Testing Steps

1. **Restart Jupyter Kernel**: Clear any cached state
   - In Jupyter: Kernel → Restart
   
2. **Run Cells in Order**:
   - Cell 0: Imports (should work even without torch)
   - Cell 1: Configuration (will use numpy if torch unavailable)
   - Cell 2: Initialization (should work with numpy fallback)

3. **Check Output**:
   - You should see messages about which diagonalization method is being used
   - If torch is unavailable, it should say "using numpy/scipy"
   - Matrix size warnings if the matrix is very large

## If Kernel Still Crashes

### Option 1: Install PyTorch Properly
```bash
# In Anaconda Prompt
conda install pytorch cpuonly -c pytorch
# OR for GPU
conda install pytorch cudatoolkit -c pytorch
```

### Option 2: Reduce Problem Size
If memory is the issue, try reducing the basis size:
```python
X010_173 = MoleculeLevels.initialize_state(
    "YbOH",
    "173",
    "X010",
    [1, 1],  # Reduce from [1, 2] to [1, 1]
    M_values="pos",  # Use "pos" instead of "all" to reduce basis
    I=[5 / 2, 1 / 2],
    S=1 / 2,
    round=8,
    P_values=[1 / 2, 3 / 2],
)
```

### Option 3: Use Numpy Explicitly
Force numpy method during initialization by modifying `Energy_Levels.py` line 135:
```python
self.eigensystem(0,1e-6, method='numpy', order=False)
```
(This is already the default now)

### Option 4: Check Memory
- Check available RAM: Large matrices (e.g., 10000x10000) require significant memory
- Close other applications to free up memory
- Consider using a machine with more RAM

## Expected Behavior

### With PyTorch Installed:
- Cell 0: "PyTorch X.X.X is available"
- Cell 1: "Using torch device: cpu" (or cuda)
- Cell 2: Initialization works, may show matrix size warning

### Without PyTorch:
- Cell 0: "Warning: PyTorch is not installed. Using numpy/scipy for diagonalization."
- Cell 1: "Torch not available, using numpy/scipy for diagonalization"
- Cell 2: Initialization works using numpy

## Additional Notes

- The code now defaults to numpy during initialization for stability
- Torch can still be used for later calculations if available
- All error messages now provide helpful information about what went wrong
- Memory checks warn if matrices are very large (>10000x10000)

## Next Steps

1. Restart your Jupyter kernel
2. Run the cells in order
3. Check the output messages to see which method is being used
4. If it still crashes, try reducing the problem size (N_range, M_values)
5. If problems persist, check system memory and PyTorch installation

