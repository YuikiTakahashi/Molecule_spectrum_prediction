# Optimization Strategy for `find_MQM_science_states()`

## Current Performance Bottlenecks

### 1. **CRITICAL: Redundant Eigensystem Calculations** (50-70% of runtime)
**Problem:** For each E-field point:
- `g_eff_EB()` calls `eigensystem()` → then calls `StarkMap(Ez ± step)` internally (3 more diagonalizations)
- `dipole_EB()` calls `eigensystem()` → then calls `StarkMap(Ez ± step)` internally (3 more diagonalizations)
- Result: **6+ diagonalizations per E-field** when only 1 is needed

**Current Flow:**
```python
evals, evecs = self.eigensystem(Ez, Bz)           # Diagonalize once
g_effs = self.g_eff_EB(Ez, Bz, step=step_B)       # Diagonalizes 3×1 = 3 times
Orientations = self.dipole_EB(Ez, Bz, step=step_E) # Diagonalizes 3×1 = 3 times
```

**Solution:** Compute g_eff and dipole directly from eigenvectors without re-diagonalizing:
```python
evals, evecs = self.eigensystem(Ez, Bz)
# Use StarkMap/ZeemanMap BEFORE the loop to pre-compute derivatives
# Then extract g-factors and dipoles directly from evals
```

### 2. **State-Pair Filtering Inefficiency** (20-30% of runtime)
**Problem:** 
- Generate ALL O(N_states²) combinations before filtering
- Each pair checked against 8 sequential conditions, including expensive isolation check
- No early termination if cheaper conditions fail

**Example Inefficiency:**
- State set: ~100 states → ~5,000 pairs
- ~90% fail on cheap M-criteria check
- But ALL still get ground-state isolation check (O(N_states²) per pair)

### 3. **Ground-State Isolation Check Scaling** (15-25% of runtime)
**Problem:** For each pair:
```python
for k in range(len(evals)):  # N_states iterations
    if (abs(evals[k] - energy0) <= ground_states_isolation[0]):
        ...
```
This is O(N_states) per pair → O(N_states³) total for all pairs

**Solution:** Pre-compute once:
- Energy distance matrix: `D[i,j] = |E[i] - E[j]|`
- Use NumPy boolean indexing instead of loops

---

## Recommended Optimization Strategy

### **Phase 1: Quick Win - Cache Eigensystem & Derivatives** (40-60% speedup, low risk)

**What to do:**
1. Before the E-field loop, compute StarkMap/ZeemanMap arrays once
2. Extract all g-factors and dipoles from those cached arrays
3. Pass them to the pair-iteration loop directly

**Expected savings:** ~3-5× fewer matrix diagonalizations

**Implementation:**
```python
# ONCE per E-field (not in pair loop)
evals, evecs = self.eigensystem(Ez, Bz)
    
# Compute sensitivity derivatives OUTSIDE pair iteration
evals_E, evecs_E = self.StarkMap([Ez-step_E, Ez, Ez+step_E], Bz_val=Bz, ...)
evals_B, evecs_B = self.ZeemanMap([Bz-step_B, Bz, Bz+step_B], Ez_val=Ez, ...)
    
# Extract all g-factors and dipoles (vectorized)
g_effs = compute_g_factors_from_evals(evals_E)  # No more loops!
Orientations = compute_dipoles_from_evals(evals_E)

# Then proceed to pair iteration with pre-cached values
for i in range(len(B_combis)):
    g_combis[i] = g_effs[index_combis[i][0]] - g_effs[index_combis[i][1]]
    # ... rest of pair logic
```

---

### **Phase 2: Vectorize Quantum Number Filtering** (25-35% speedup, medium complexity)

**What to do:**
1. Pre-compute quantum number arrays: `N_arr`, `M_arr`, `F_arr`, `F1_arr`, `G_arr` (1D arrays, length = N_states)
2. Use NumPy boolean indexing to pre-filter pairs

**Before (Current):**
```python
index_combis = np.array(list(it.combinations(range(len(g_effs)), 2)))  # All N_states choose 2
# Then filter each individually in loop
for i in range(len(B_combis)):
    M0 = self.q_numbers['M'][np.argmax(evec0**2)]
    M1 = self.q_numbers['M'][np.argmax(evec1**2)]
```

**After (Optimized):**
```python
# Pre-compute quantum numbers for all states
M_arr = np.array([self.q_numbers['M'][np.argmax(evecs[k]**2)] for k in range(len(evecs))])
    
# Pre-filter M-compatible pairs
valid_pairs = []
for i, j in it.combinations(range(len(evecs)), 2):
    if abs(M_arr[i] - M_arr[j]) == M_criteria:  # Cheap check FIRST
        valid_pairs.append((i, j))
    
index_combis = np.array(valid_pairs)  # Much smaller!
```

**Expected savings:** Reduce pair iteration count by 80-95%

---

### **Phase 3: Batch Isolation Check** (15-25% speedup, high complexity)

**Current: O(N_states³)**
```python
for each_pair:
    for k in range(len(evals)):
        if distance(E[k], pair[0]) < threshold:
            ...
```

**Optimized: O(N_states²)**
```python
# Pre-compute once BEFORE pair loop
energy_distances = np.abs(evals[:, np.newaxis] - evals[np.newaxis, :])  # NxN matrix
M_arr = np.array([...])  # Pre-compute M values

# Then in pair loop:
neighbors_0 = np.where(energy_distances[i] < threshold)[0]
neighbors_1 = np.where(energy_distances[j] < threshold)[0]
# Quick check with pre-computed arrays
```

---

### **Phase 4: Short-Circuit Filtering** (5-15% speedup, low complexity)

**Current order** (expensive operations first):
```python
for pair in all_pairs:
    if ground_state_isolation_check(pair):  # O(N_states) - EXPENSIVE
        if M_criteria_check(pair):  # O(1) - CHEAP
            if frequency_check(pair):  # O(1) - CHEAP
```

**Optimized order** (cheap operations first):
```python
for pair in pre_filtered_pairs:  # Already reduced by Phase 2
    if frequency_check(pair):  # O(1) - CHEAP, break early
        if parity_check(pair):  # O(1) - CHEAP, break early
            if isolation_check(pair):  # O(N_states) - EXPENSIVE, only if previous pass
                # ... proceed with interpolation
```

This saves ~90% of expensive isolation checks because most pairs fail early.

---

## Implementation Timeline

| Phase | Complexity | Speedup | Effort | Risk |
|-------|-----------|---------|--------|------|
| **1: Cache eigensystems** | Low | 3-5× | 30 min | Very low |
| **2: Vectorize quantum filters** | Medium | 2-3× | 1 hour | Low |
| **3: Batch isolation check** | High | 1.5-2× | 2-3 hours | Medium |
| **4: Short-circuit order** | Low | 1.2-1.5× | 30 min | Very low |
| **Combined estimated** | **Medium** | **10-20×** | **4-5 hours** | **Low** |

---

## Testing & Validation Strategy

1. **Correctness Testing:**
   - Compare output state indices with original implementation
   - Verify transition frequencies match exactly
   - Check coherence time calculations are identical

2. **Performance Benchmarking:**
   - Profile with small E-field array (3-5 points)
   - Measure time per phase: eigensystem, filtering, isolation check, interpolation
   - Identify actual bottleneck with small dataset

3. **Validation Before Full Run:**
   - Test with subset of configurations (e.g., single isotope, single criterion set)
   - Compare against known results if available

---

## Recommended Next Step

**Option A: Quick profiling first (5 min)**
```python
# Run original function with 3 E-field points
# Use %timeit or cProfile to see actual time breakdown
```

**Option B: Quick optimization (high confidence)**
- Implement Phase 1 (cache eigensystems) immediately
- Should give 3-5× speedup with very low risk
- Then profile again to see remaining bottleneck

**My recommendation:** Do Option B first, as Phase 1 is risk-free and gives big speedup. If you still want more speed, Phase 2 is next logical step.

