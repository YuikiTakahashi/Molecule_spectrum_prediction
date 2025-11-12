# Code Explanation and Fixes

## What This Code Does

This codebase is designed to predict molecular spectra for molecules like YbOH (Ytterbium Hydroxide). It uses quantum mechanical calculations to:

1. **Build Hamiltonian matrices** for different molecular states and isotopes
2. **Diagonalize these matrices** to find energy eigenvalues and eigenvectors
3. **Calculate transition frequencies** between energy levels
4. **Fit parameters** to match experimental data
5. **Optimize parameters** using least-squares fitting

## Main Components

### 1. `Energy_Levels.py`
- **`MoleculeLevels` class**: Main class that represents a molecular state
- **`initialize_state()` method**: Creates a `MoleculeLevels` object for a specific molecule, isotope, and state
- **`eigensystem()` method**: Diagonalizes the Hamiltonian matrix to find energy levels
- **`diagonalize()` function**: Performs matrix diagonalization using torch, numpy, or scipy

### 2. `molecule_library_class.py`
- **`Molecule_Library` class**: Contains all the matrix elements, Hamiltonian builders, and quantum number builders for different molecules
- Manages the relationship between different molecular states and their properties

### 3. `molecule_parameters.py`
- Contains all the physical parameters (rotation constants, hyperfine constants, etc.) for different molecules and isotopes
- Parameters are stored in dictionaries indexed by molecule and isotope-state (e.g., '173X010' for YbOH isotope 173, X state, vibrational level 010)

### 4. `173_prediction_Nov2025.ipynb`
- Main notebook that:
  - Initializes a molecular state (YbOH-173 X010)
  - Sets up parameter fitting
  - Calculates transition frequencies
  - Compares with experimental data
  - Performs optimization to fit parameters

## The Problem

When running the notebook, you encountered an error during the `initialize_state()` call. The main issues were:

1. **Missing `torch` module**: PyTorch was not installed in your Anaconda environment
2. **Tensor conversion bug**: The `diagonalize()` function in `Energy_Levels.py` was calling `.numpy()` directly on tensors, which can fail in newer PyTorch versions or when tensors are on GPU
3. **Missing `order` parameter handling**: The `order` parameter in `diagonalize()` was commented out, causing issues when ordering was requested

## The Fixes

### 1. Created `requirements.txt`
Added a requirements file listing all necessary dependencies:
- numpy, pandas, sympy
- torch (PyTorch)
- scipy, matplotlib, seaborn
- scikit-learn, jupyter

### 2. Fixed `Energy_Levels.py`
- Changed `.numpy()` to `.detach().cpu().numpy()` in both `diagonalize()` and `diagonalize_batch()` functions
- This ensures safe tensor-to-numpy conversion regardless of device or gradient requirements
- Enabled the `order` parameter handling that was previously commented out

### 3. Improved Notebook Cell 1
- Added better comments explaining the patching mechanism
- Added proper error handling and fallback behavior
- Added a confirmation message when patching succeeds

## How to Use

1. **Install dependencies**:
   ```bash
   conda install pytorch cpuonly -c pytorch  # For CPU
   # OR
   conda install pytorch cudatoolkit -c pytorch  # For GPU
   pip install -r requirements.txt
   ```

2. **Run the notebook**:
   - Open `173_prediction_Nov2025.ipynb` in Jupyter Notebook
   - Run cells in order
   - The first cell imports all modules
   - The second cell configures torch and patches diagonalization functions
   - The third cell initializes the molecular state

## Key Parameters

When initializing a state with `MoleculeLevels.initialize_state()`:
- **molecule**: "YbOH", "CaOH", or "YbF"
- **isotope**: "173", "174", "171", etc.
- **state**: "X010", "X000", "A000", etc. (electronic state + vibrational level)
- **N_range**: [1, 2] - range of rotational quantum numbers
- **M_values**: "all" - all M quantum number values
- **I**: [5/2, 1/2] - nuclear spin values for Yb and H
- **S**: 1/2 - electron spin
- **P_values**: [1/2, 3/2] - parity values
- **round**: 8 - rounding precision for eigenvalues/eigenvectors

## Next Steps

1. **Install PyTorch** in your Anaconda environment
2. **Run the notebook** cells in order
3. **Update experimental data paths** in cell 4 if you have experimental data
4. **Adjust parameters** as needed for your specific use case

## Troubleshooting

If you still encounter errors:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
3. Verify the molecular state and isotope combination exists in `molecule_parameters.py`
4. Check that all required parameters are defined for your molecular state

## Notes

- The code uses PyTorch for faster matrix diagonalization (especially on GPU)
- If PyTorch is not available, it falls back to numpy/scipy
- The `diagonalize()` function is patched in the notebook to use GPU/CPU device management
- All calculations are done in MHz units
- Electric fields are in V/cm, magnetic fields are in Gauss

