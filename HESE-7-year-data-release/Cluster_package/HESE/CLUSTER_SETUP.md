# Cluster Setup Instructions

## Files Included

This package contains the minimal set of files needed to run HESE fits on the cluster:

### Core Python Files
- `HESE_fit.py` - Main fitting script
- `weighter_original.py` - Weight calculations
- `data_loader.py` - Data loading
- `det_sys_weights.py` - Detector systematic corrections
- `likelihood.py` - Likelihood calculations
- `autodiff.py` - Automatic differentiation
- `binning.py` - Analysis binning
- `scan_2d_grid.py` - 2D grid scan script
- `aggregate_grid_results.py` - Result aggregation

### Data Files
- `resources/` - Contains data files and splines

## Setup on Cluster

1. **Place files in your cluster repo**:
   ```bash
   # If you have a repo structure like:
   # /path/to/repo/
   #   ├── nuSIprop/
   #   └── HESE/  (put files here)
   
   cp -r HESE/* /path/to/repo/HESE/
   ```

2. **Update nuSIprop path in HESE_fit.py**:
   
   The current code looks for nuSIprop at `../../nuSIprop` relative to HESE_fit.py.
   
   **If your structure is:**
   ```
   repo/
     ├── nuSIprop/
     └── HESE/
         └── HESE_fit.py
   ```
   
   Then update the path to `../nuSIprop`:
   ```bash
   python update_nusiprop_path.py HESE_fit.py --nusiprop_path ../nuSIprop
   ```
   
   **Or if nuSIprop is at the same level as the repo:**
   ```
   parent_dir/
     ├── nuSIprop/
     └── repo/
         └── HESE/
             └── HESE_fit.py
   ```
   
   Then use:
   ```bash
   python update_nusiprop_path.py HESE_fit.py --nusiprop_path ../../nuSIprop
   ```
   
   **Or manually edit HESE_fit.py line ~26** if needed.

3. **Test the setup**:
   ```bash
   cd /path/to/repo/HESE
   python HESE_fit.py --model spl --fix_epsilon_dom
   ```

4. **Run grid scan**:
   ```bash
   # Edit submit_grid_scan.sh with correct paths
   sbatch submit_grid_scan.sh
   ```

## Dependencies

- Python packages: numpy, scipy, pandas (for aggregate script)
- nuSIprop library (should be in parent directory or adjust path)
- PHOTOSPLINE (for spline interpolation in det_sys_weights.py)

## Notes

- All paths in the scripts are relative, so they should work as long as the
  directory structure is preserved.
- The `resources/` directory must be in the same directory as `HESE_fit.py`.
