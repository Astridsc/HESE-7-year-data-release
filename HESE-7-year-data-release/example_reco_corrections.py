"""
Example script showing how to add reconstruction correction nuisance parameters
to handle datasets with different reconstruction methods.

This demonstrates how to:
1. Add nuisance parameters for energy scale and zenith shift
2. Load the new dataset (164 events) or test with original (102 events)
3. Pass the original data to calcLLH for re-binning with corrections

Usage:
    # Test with original 102-event dataset (should give same results as HESE_fit.py)
    python example_reco_corrections.py --test_original
    
    # Test with new 164-event dataset
    python example_reco_corrections.py --data_file ./resources/data/HESE_data_164.json
"""

import numpy as np
import argparse
import data_loader
import binning
import weighter
import likelihood
from scipy.optimize import fmin_l_bfgs_b
import time

parser = argparse.ArgumentParser(description="Test reconstruction corrections")
parser.add_argument("--data_file", type=str, default=None,
                    help="Path to data file (default: original 102-event dataset)")
parser.add_argument("--test_original", action="store_true",
                    help="Test with original dataset (reco corrections should be ~1.0, 0.0)")
parser.add_argument("--fix_reco_energy_scale", action="store_true",
                    help="Fix energy scale correction at 1.0")
parser.add_argument("--fix_reco_zenith_shift", action="store_true",
                    help="Fix zenith shift correction at 0.0")
args = parser.parse_args()

# Load dataset
if args.test_original or args.data_file is None:
    # Use original 102-event dataset for testing
    data_file = "./resources/data/HESE_data.json"
    print("Testing with original 102-event dataset")
    print("Expected: reco_energy_scale ≈ 1.0, reco_zenith_shift ≈ 0.0")
else:
    data_file = args.data_file
    print(f"Testing with dataset: {data_file}")

data_new = data_loader.load_data(data_file)
print(f"Loaded {len(data_new)} events")

# Load MC (unchanged)
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
mc = data_loader.load_mc(mc_filenames)

# Bin MC (unchanged)
sorted_mc, mc_bin_slices = binning.bin_data(mc)

# Add reconstruction correction parameters
parameter_names = [
    "cr_delta_gamma",
    "nunubar_ratio",
    "anisotropy_scale",
    "astro_gamma",
    "astro_norm",
    "conv_norm",
    "epsilon_dom",
    "epsilon_head_on",
    "muon_norm",
    "kpi_ratio",
    "prompt_norm",
    "beta",
    "cutoff_energy",
    # NEW: Reconstruction correction parameters
    "reco_energy_scale",  # Energy scale factor (1.0 = no correction)
    "reco_zenith_shift",  # Zenith shift in radians (0.0 = no correction)
]

# Initial parameter values
params = np.array([
    -0.05,   # cr_delta_gamma
    1.0,     # nunubar_ratio
    1.0,     # anisotropy_scale
    2.5,     # astro_gamma
    1.0e-18, # astro_norm
    1.0,     # conv_norm
    0.99,    # epsilon_dom
    0.0,     # epsilon_head_on
    1.0,     # muon_norm
    1.0,     # kpi_ratio
    0.0,     # prompt_norm
    0.0,     # beta
    0.0,     # cutoff_energy
    # NEW: Reconstruction corrections
    1.0,     # reco_energy_scale (no correction by default)
    0.0,     # reco_zenith_shift (no correction by default)
])

# Priors (add priors for new parameters)
priors = [
    (-0.05, 0.05, -np.inf, np.inf),  # cr_delta_gamma
    (1.0, 0.1, 0.0, 2.0),            # nunubar_ratio
    (1.0, 0.2, 0.0, 2.0),            # anisotropy_scale
    (None, None, -np.inf, np.inf),   # astro_gamma
    (None, None, 0.0, np.inf),       # astro_norm
    (1.0, 0.4, 0.0, np.inf),         # conv_norm
    (0.99, 0.1, 0.8, 1.25),          # epsilon_dom
    (0.0, 0.5, -3.82, 2.18),         # epsilon_head_on
    (1.0, 0.5, 0.0, np.inf),         # muon_norm
    (1.0, 0.1, 0.0, np.inf),         # kpi_ratio
    (None, None, 0.0, np.inf),       # prompt_norm
    (None, None, -np.inf, np.inf),   # beta
    (None, None, 1e5, 1e7, "log_uniform"),  # cutoff_energy
    # NEW: Priors for reconstruction corrections
    # Energy scale: Gaussian prior centered at 1.0 (no correction)
    # Adjust sigma based on your uncertainty estimate
    (1.0, 0.05, 0.9, 1.1),           # reco_energy_scale
    # Zenith shift: Gaussian prior centered at 0.0 (no correction)
    # Adjust sigma based on your uncertainty estimate (in radians)
    (0.0, 0.1, -0.5, 0.5),          # reco_zenith_shift
]

# Which parameters to fit (set to False to fix)
is_fixed = [
    False,  # cr_delta_gamma
    False,  # nunubar_ratio
    False,  # anisotropy_scale
    False,  # astro_gamma
    False,  # astro_norm
    False,  # conv_norm
    False,  # epsilon_dom
    False,  # epsilon_head_on
    False,  # muon_norm
    False,  # kpi_ratio
    False,  # prompt_norm
    False,  # beta
    False,  # cutoff_energy
    # NEW: Fit reconstruction corrections
    False,  # reco_energy_scale
    False,  # reco_zenith_shift
]

is_fitted = [not b for b in is_fixed]

livetime = 227708167.68
weight_maker = weighter.Weighter(sorted_mc, nuSIprop=False, model="spl")

# Wrapper function for fitting
def calcLLH_fitted_func(is_fitted, params, original_data):
    def func(
        fitted_params,
        parameter_names,
        priors,
        mc_bin_slices,
        binned_data,  # Not used when apply_reco_corrections=True
        weights,
        livetime,
    ):
        params[:][is_fitted] = fitted_params
        # Pass original_data and enable reconstruction corrections
        llh, grads = likelihood.calcLLH(
            params,
            parameter_names,
            priors,
            mc_bin_slices,
            None,  # data will be computed from original_data
            weights,
            livetime,
            original_data=original_data,  # Pass original uncorrected data
            apply_reco_corrections=True,  # Enable corrections
        )
        return llh, np.array(grads[0])[is_fitted]
    return func

# Handle epsilon_dom bimodality (like in HESE_fit.py)
bounds_list = []
if args.fix_reco_energy_scale and args.fix_reco_zenith_shift:
    # If both reco corrections are fixed, don't split epsilon_dom
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    bounds_list.append(bounds)
else:
    # Split epsilon_dom boundaries
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    epsilon_dom_idx = parameter_names.index("epsilon_dom")
    bounds_low = np.copy(bounds)
    bounds_high = np.copy(bounds)
    bounds_low[epsilon_dom_idx] = [0.8, 0.99]
    bounds_high[epsilon_dom_idx] = [0.99, 1.25]
    bounds_list.append(bounds_low)
    bounds_list.append(bounds_high)

# Update is_fixed based on command-line arguments
if args.fix_reco_energy_scale:
    reco_energy_idx = parameter_names.index("reco_energy_scale")
    is_fixed[reco_energy_idx] = True
if args.fix_reco_zenith_shift:
    reco_zenith_idx = parameter_names.index("reco_zenith_shift")
    is_fixed[reco_zenith_idx] = True

is_fitted = [not b for b in is_fixed]

if np.any(is_fixed):
    print("\nFixing parameters:")
    for b, name, val in zip(is_fixed, parameter_names, params):
        if b:
            print(f"  {name} = {val}")

calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params), data_new)

# Run fits
fitted_params_list = []
llh_list = []
info_list = []

print("\nRunning fit(s)...")
start = time.time()

for bounds in bounds_list:
    fitted_params, llh, info = fmin_l_bfgs_b(
        calcLLH,
        x0=params[is_fitted],
        args=(
            parameter_names,
            priors,
            mc_bin_slices,
            None,  # Not used when apply_reco_corrections=True
            weight_maker,
            livetime,
        ),
        bounds=bounds[is_fitted],
        m=10,
        pgtol=1e-18,
        factr=1e4,
    )
    fitted_params_list.append(fitted_params)
    llh_list.append(llh)
    info_list.append(info)

end = time.time()
print(f"Fit took {end - start:.2f} seconds")

# Pick best fit
min_index = np.argmin(llh_list)
BF_fitted_params = fitted_params_list[min_index]
BF_llh = llh_list[min_index]
BF_info = info_list[min_index]

print("\n" + "="*60)
print("BEST FIT RESULTS")
print("="*60)
print(f"Best Fit -LLH: {BF_llh:.6f}")
print(f"Fit status: {BF_info['warnflag']} (0=converged, 1=max iterations, 2=other)")
print("\nBest Fit Parameters:")
BF_params = params[:]
BF_params[is_fitted] = BF_fitted_params
for param, BF_param in zip(parameter_names, BF_params):
    print(f"  {param:20s}: {BF_param:15.8e}")

# Validation checks
print("\n" + "="*60)
print("VALIDATION CHECKS")
print("="*60)

# Check if reco corrections are reasonable
reco_energy_idx = parameter_names.index("reco_energy_scale")
reco_zenith_idx = parameter_names.index("reco_zenith_shift")
reco_energy_val = BF_params[reco_energy_idx]
reco_zenith_val = BF_params[reco_zenith_idx]

print(f"\nReconstruction Corrections:")
print(f"  Energy scale: {reco_energy_val:.6f} (1.0 = no correction)")
print(f"  Zenith shift: {reco_zenith_val:.6f} rad (0.0 = no correction)")

if args.test_original:
    print("\n✓ Testing with original dataset:")
    if abs(reco_energy_val - 1.0) < 0.01:
        print("  ✓ Energy scale is ~1.0 (expected)")
    else:
        print(f"  ⚠ Energy scale is {reco_energy_val:.6f} (expected ~1.0)")
    
    if abs(reco_zenith_val) < 0.01:
        print("  ✓ Zenith shift is ~0.0 (expected)")
    else:
        print(f"  ⚠ Zenith shift is {reco_zenith_val:.6f} (expected ~0.0)")

# Check if parameters are within bounds
print("\nParameter bounds check:")
all_in_bounds = True
for param_name, param_val, prior in zip(parameter_names, BF_params, priors):
    low, high = prior[2], prior[3]
    if param_val < low or param_val > high:
        print(f"  ⚠ {param_name} = {param_val:.6e} is outside bounds [{low}, {high}]")
        all_in_bounds = False
if all_in_bounds:
    print("  ✓ All parameters within prior bounds")

# Check fit convergence
if BF_info['warnflag'] == 0:
    print("\n✓ Fit converged successfully")
else:
    print(f"\n⚠ Fit warning flag: {BF_info['warnflag']}")

print("\n" + "="*60)

