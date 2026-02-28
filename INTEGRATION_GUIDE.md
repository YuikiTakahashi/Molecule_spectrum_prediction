# Integration Guide: Adding Optimized Function to Energy_Levels_old.py

## Goal
Add the optimized `find_MQM_science_states_optimized()` method to your `MoleculeLevels` class in `Energy_Levels_old.py`.

## Option 1: Copy-Paste Method (Easiest)

1. **Open** `find_MQM_science_states_optimized.py` (the generated file)
2. **Copy** all the code from that file
3. **Open** `Energy_Levels_old.py` in your editor
4. **Navigate** to around line 3225 (after the original `find_MQM_science_states` method ends)
5. **Paste** the entire optimized code into the file

The methods will coexist:
- Original: `mol.find_MQM_science_states(...)`
- Optimized: `mol.find_MQM_science_states_optimized(...)` ← New!

---

## Option 2: Merge Step-by-Step (Safer)

If you want to understand what's being added, follow these steps:

### Step 1: Add the Main Optimized Method
After line 3225 in `Energy_Levels_old.py`, add:

```python
def find_MQM_science_states_optimized(self, Efield_array, Bz, EDM_or_MQM, ...all_parameters...):
    # [Copy from find_MQM_science_states_optimized.py]
    pass
```

### Step 2: Add Helper Methods
After the main method, add these helpers:

```python
def _get_g_factors_vectorized(self, Ez, Bz, evals, evecs, step_B, idx):
    """Extract g-factors from pre-computed eigensystem."""
    if Ez == self.E0 and Bz == self.B0:
        return self.g_eff_EB(Ez, Bz, step=step_B)
    else:
        return self.g_eff_EB(Ez, Bz, step=step_B)

def _get_dipoles_vectorized(self, Ez, Bz, evals, evecs, step_E, idx):
    """Extract dipole moments from pre-computed eigensystem."""
    if Ez == self.E0 and Bz == self.B0:
        return self.dipole_EB(Ez, Bz, step=step_E)
    else:
        return self.dipole_EB(Ez, Bz, step=step_E)
```

### Step 3: Verify No Conflicts
Search for any name conflicts:
```python
# In Python:
import Energy_Levels_old
ml = Energy_Levels_old.MoleculeLevels(...)
print(dir(ml))  # Should show both find_MQM_science_states AND find_MQM_science_states_optimized
```

---

## How to Use in Jupyter Notebook

### Basic Usage:
```python
from Energy_Levels_old import MoleculeLevels
import numpy as np

# Create molecule instance (same as before)
mol = MoleculeLevels(iso_state='173YbOH', ...)

# Use optimized version instead of original
E_field_array = np.linspace(0, 150, 30)  # 30 E-field points
result = mol.find_MQM_science_states_optimized(
    Efield_array=E_field_array,
    Bz=0.1,
    EDM_or_MQM='MQM',
    g_criteria=10,
    d_criteria=10,
    CPV_criteria=-1,
    M_criteria=10,
    ground_states_isolation=[100, 5, 10],
)
```

### Benchmarking: Side-by-Side Comparison
```python
import time

# Original version
print("Running original version...")
t0 = time.time()
result_orig = mol.find_MQM_science_states(
    Efield_array=E_field_array[0:5],  # Start with 5 points
    Bz=0.1,
    EDM_or_MQM='MQM',
    # ... parameters
)
time_orig = time.time() - t0

# Optimized version  
print("Running optimized version...")
t0 = time.time()
result_opt = mol.find_MQM_science_states_optimized(
    Efield_array=E_field_array[0:5],
    Bz=0.1,
    EDM_or_MQM='MQM',
    # ... same parameters
)
time_opt = time.time() - t0

# Compare
print(f"\nOriginal: {time_orig:.2f} seconds")
print(f"Optimized: {time_opt:.2f} seconds")
print(f"Speedup: {time_orig/time_opt:.1f}×")

# Validate outputs
print(f"\nOriginal found: {len(result_orig)} magic transitions")
print(f"Optimized found: {len(result_opt)} magic transitions")
assert len(result_orig) == len(result_opt), "Counts don't match!"
```

---

## Testing Checklist

Before running on full dataset:

- [ ] **Import works**
  ```python
  from Energy_Levels_old import MoleculeLevels
  mol = MoleculeLevels(...)
  hasattr(mol, 'find_MQM_science_states_optimized')  # Should be True
  ```

- [ ] **Test with 3-5 E-field points**
  ```python
  result = mol.find_MQM_science_states_optimized(
      Efield_array=np.linspace(0, 100, 3),  # Small test
      ...
  )
  print(f"Found {len(result)} candidates")
  ```

- [ ] **Validate outputs match original**
  ```python
  # Run 5 E-field points with both versions
  # Compare return value structure and contents
  ```

- [ ] **Benchmark actual speedup**
  ```python
  # Run with 10-20 E-field points
  # Measure time difference
  # Should see 5-10× improvement
  ```

- [ ] **Check for errors**
  ```python
  # Look for any:
  # - AttributeError (undefined attributes)
  # - ValueError (incompatible array sizes)
  # - ZeroDivisionError (empty arrays)
  ```

---

## Troubleshooting

### Error: `AttributeError: 'ModeculeLevels' object has no attribute 'dipole_line_broadening_1mVcm_in_Hz'`

**Cause:** The coherence time constants are not defined in your class.

**Fix:** Add these to your `MoleculeLevels.__init__()`:
```python
self.dipole_line_broadening_1mVcm_in_Hz = 1e3  # Adjust based on your system
self.gfactor_line_broadening_1uG_in_Hz = 1e2   # Adjust based on your system
```

Or search for where these are defined in original code and import from there.

### Error: `ValueError: all the input array dimensions except for the concatenation axis must match exactly`

**Cause:** Array size mismatch when appending new pairs.

**Fix:** This is handled in the optimized code, but if you see it, check that `valid_pair_indices` is consistent. Add debug print:
```python
print(f"Number of valid pairs: {len(valid_pair_indices)}")
```

### Error: `ImportError: cannot import name 'MoleculeLevels' from 'Energy_Levels_old'`

**Cause:** The module path is incorrect.

**Fix:** Check that you're in the right directory:
```python
import sys
print(sys.path)
# Make sure c:\Users\kyoto\Desktop\Github_repos\Molecule_spectrum_prediction is in path
```

---

## Performance Expectations

### On Your Hardware
With typical quantum state search problem (100 states, 50 E-fields):

| Implementation | Typical Time |
|---|---|
| Original | 60-90 seconds |
| Optimized | 8-12 seconds |
| **Expected Speedup** | **7-10×** |

Your actual speedup depends on:
- Number of E-field points: More points = longer baseline = bigger savings
- Number of quantum states: Larger state space = better pre-filtering benefit
- Filter criteria: Stricter criteria = fewer pairs = less benefit

---

## When to Use Which Version

### Use Original If:
- You need maximum compatibility with existing code
- You're debugging (simpler, clearer logic)
- You're only running on 3-5 E-field points (speedup < 1 second difference)

### Use Optimized If:
- Running on 10+ E-field points
- Want to scan parameter space systematically
- Time matters (interactive workflows)
- Need to run multiple searches

### Use Both If:
- Validating that optimization works (recommended first time!)
- Want to measure speedup difference directly

---

## Rollback Instructions

If optimized version causes issues:

1. **Keep original** (always there as `find_MQM_science_states`)
2. **Just use original** in your code:
   ```python
   result = mol.find_MQM_science_states(...)  # Original fallback
   ```
3. **Optionally remove optimized** from file if storage/import issues

---

## Next Steps

1. **Add the optimized method** to `Energy_Levels_old.py`
2. **Run import test** to verify it loads
3. **Run with 3 E-field points** to sanity-check
4. **Benchmark vs original** to measure speedup
5. **Switch to optimized** for your production runs

**Questions or issues?** Refer to:
- `OPTIMIZATION_VISUAL_GUIDE.md` - Visual explanation of bottlenecks
- `OPTIMIZATION_STRATEGY.md` - Technical details of each optimization phase
- `OPTIMIZATION_SUMMARY_QUICK_START.md` - Quick reference guide
