"""
This file runs a fit of the data over a single power-law astrophysical flux
model. The initial value of the fits can be modified within the file or through
command line arguments. One can also choose to fix certain parameters in the
fit, where the fixed value is kept at the initial value
"""

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import argparse
import time
import gc
#import weighter
import weighter_original
import binning
import data_loader
import autodiff
import likelihood
import det_sys_weights
import sys
import os
import os.path
# Add nuSIprop to path (../../nuSIprop from this file's location)
base_path = os.path.dirname(os.path.abspath(__file__))
nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop'))
if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
import nuSIprop

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cr_delta_gamma", default=-0.05, type=float, help="set initial cosmic ray slope"
)
parser.add_argument(
    "--nunubar_ratio",
    default=1.0,
    type=float,
    help="set initial neutrino/antineutrino ratio",
)
parser.add_argument(
    "--anisotropy_scale",
    default=1.0,
    type=float,
    help="set initial ice anisotropy scale",
)
parser.add_argument(
    "--astro_gamma",
    default=2.5,
    type=float,
    help="set initial astrophysical spectral index",
)
parser.add_argument(
    "--astro_norm",
    default=6.0,
    type=float,
    help="set initial astrophysical six-neutrino flux normalization",
)
parser.add_argument(
    "--conv_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric conventional neutrino flux normalization",
)
parser.add_argument(
    "--epsilon_dom",
    default=0.99,
    type=float,
    help="set initial DOM absolute energy scale",
)
parser.add_argument(
    "--epsilon_head_on",
    default=0.0,
    type=float,
    help="set initial DOM angular response",
)
parser.add_argument(
    "--muon_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric muon flux normalization",
)
parser.add_argument(
    "--kpi_ratio",
    default=1.0,
    type=float,
    help="set initial kaon/pion ratio correction",
)
parser.add_argument(
    "--prompt_norm",
    default=1.0,
    type=float,
    help="set initial atmospheric prompt neutrino flux normalization",
)


parser.add_argument(
    "--model", default="spl", type=str,
                    choices=["spl", "cutoff", "nusiprop"],
                    help="astrophysical flux model: 'spl' (single power law), 'cutoff' (exponential cutoff), 'nusiprop' (nuSIprop)")
parser.add_argument(
    "--cutoff_energy", default=1e5, type=float,
                    help="set initial cutoff energy parameter for exponential cutoff")
parser.add_argument(
    "--fix_cutoff_energy", action="store_true",
                    help="fix cutoff energy parameter for exponential cutoff in fit")
parser.add_argument("--Mphi", default=5.0, type=float,
                    help="set initial Mphi parameter for nuSIprop (in GeV)")
parser.add_argument("--g", default=0.1, type=float,
                    help="set initial g parameter for nuSIprop")
parser.add_argument("--mntot", default=0.1, type=float,
                    help="set initial mntot parameter for nuSIprop")



parser.add_argument(
    "--fix_cr_delta_gamma", action="store_true", help="fix cosmic ray slope in fit"
)
parser.add_argument(
    "--fix_nunubar_ratio",
    action="store_true",
    help="fix neutrino/antineutrino ratio in fit",
)
parser.add_argument(
    "--fix_anisotropy_scale",
    action="store_true",
    help="fix ice anisotropy scale in fit",
)
parser.add_argument(
    "--fix_astro_gamma",
    action="store_true",
    help="fix astrophysical spectral index in fit",
)
parser.add_argument(
    "--fix_astro_norm",
    action="store_true",
    help="fix astrophysical six-neutrino flux normalization in fit",
)
parser.add_argument(
    "--fix_conv_norm",
    action="store_true",
    help="fix atmospheric conventional neutrino flux normalization in fit",
)
parser.add_argument(
    "--fix_epsilon_dom",
    action="store_true",
    help="fix DOM absolute energy scale in fit",
)
parser.add_argument(
    "--fix_epsilon_head_on", action="store_true", help="fix DOM angular response in fit"
)
parser.add_argument(
    "--fix_muon_norm",
    action="store_true",
    help="fix atmospheric muon flux normalization in fit",
)
parser.add_argument(
    "--fix_kpi_ratio", action="store_true", help="fix kaon/pion ratio correction in fit"
)
parser.add_argument(
    "--fix_prompt_norm",
    action="store_true",
    help="fix atmospheric prompt neutrino flux normalization in fit",
)
parser.add_argument("--fix_Mphi", action="store_true",
                    help="fix Mphi parameter in fit (nuSIprop)")
parser.add_argument("--fix_g", action="store_true",
                    help="fix g parameter in fit (nuSIprop)")
parser.add_argument("--fix_mntot", action="store_true",
                    help="fix mntot parameter in fit (nuSIprop)")

args = parser.parse_args()

livetime = 227708167.68
#livetime = 12 * 365 * 24 * 3600

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
    "cutoff_energy",
]

# Add nuSIprop parameters if using nuSIprop model
if args.model == "nusiprop":
    parameter_names.extend(["Mphi", "g", "mntot"])

params = [
    args.cr_delta_gamma,
    args.nunubar_ratio,
    args.anisotropy_scale,
    args.astro_gamma,
    args.astro_norm,
    args.conv_norm,
    args.epsilon_dom,
    args.epsilon_head_on,
    args.muon_norm,
    args.kpi_ratio,
    args.prompt_norm,
    args.cutoff_energy,
]

# Add nuSIprop parameters if using nuSIprop model
if args.model == "nusiprop":
    params.extend([args.Mphi, args.g, args.mntot])  # Mphi in GeV

params = np.array(params)

# Priors used in the fit. Each parameter has either a Gaussian or uniform prior.
# Zeroth column is the mean for a Gaussian Prior, None for a uniform prior
# First column is the standard deviation for a Gaussian Prior, None for a
# uniform prior
# Second column is the lower bound
# Third column is the upper bound
priors = [
    (0.0, 0.05, -np.inf, np.inf), # cr_delta_gamma, Gaussian prior
    (1.0, 0.1, 0.0, 2.0), # nunubar_ratio, Gaussian prior
    (1.0, 0.2, 0.0, 2.0), # anisotropy_scale, Gaussian prior
    (None, None, -np.inf, np.inf), # astro_gamma, Uniform prior
    (None, None, 0.0, np.inf), # astro_norm, Uniform prior
    (1.0, 0.4, 0.0, np.inf), # conv_norm, Gaussian prior
    (0.99, 0.1, 0.8, 1.25), # epsilon_dom, Gaussian prior
    (0.0, 0.5, -3.82, 2.18), # epsilon_head_on, Gaussian prior
    (1.0, 0.5, 0.0, np.inf), # muon_norm, Gaussian prior
    (1.0, 0.1, 0.0, np.inf), # kpi_ratio, Gaussian prior
    (None, None, 0.0, np.inf), # prompt_norm, Uniform prior
    (None, None, 1e5, 1e7, "log_uniform"), # cutoff_energy, log-uniform prior
]

# Add nuSIprop priors if using nuSIprop model
if args.model == "nusiprop":
    priors.extend([
        (None, None, 0.03, 100, "log_uniform"),  # Mphi: log-uniform in GeV
        (None, None, 1e-4, 1.0, "log_uniform"),  # g: log-uniform
        (None, None, 0.06, 0.15),                # mntot: uniform
    ])

# Check that all initial parameters are within prior bounds
for param_name, param, prior in zip(parameter_names, params, priors):
    param_min = prior[2]
    param_max = prior[3]
    if param < param_min or param > param_max:
        error_message = (
            "Given value for {}, {}, is outside of prior range [{},{}]".format(
                param_name, param, param_min, param_max
            )
        )
        raise ValueError(error_message)

# is_fixed dictates what parameters will be kept fixed during the fit. By
# default all values are set to False.
is_fixed = [
    args.fix_cr_delta_gamma,
    args.fix_nunubar_ratio,
    args.fix_anisotropy_scale,
    args.fix_astro_gamma,
    args.fix_astro_norm,
    args.fix_conv_norm,
    args.fix_epsilon_dom,
    args.fix_epsilon_head_on,
    args.fix_muon_norm,
    args.fix_kpi_ratio,
    args.fix_prompt_norm,
    args.fix_cutoff_energy,
]

# Add nuSIprop fix flags if using nuSIprop model
if args.model == "nusiprop":
    is_fixed.extend([args.fix_Mphi, args.fix_g, args.fix_mntot])

if np.any(is_fixed):
    print("Fixing parameters")
    for b, name, val in zip(is_fixed, parameter_names, params):
        if b:
            print(name + " = ", val)

is_fitted = [not b for b in is_fixed]

# Load MC and data file, and return an array of events within energy and length
# bounds.
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
mc = data_loader.load_mc(mc_filenames)
data = data_loader.load_data("./resources/data/HESE_data.json")
#data = data_loader.load_data("./resources/data/HESE12_data.json")

# bin_data takes an MC/data numpy array as input, and returns
# 0: the events rearranged such that events are grouped by analysis bins.
# 1: the list of bin slices for each analysis bin.
sorted_mc, mc_bin_slices = binning.bin_data(mc)
sorted_data, data_bin_slices = binning.bin_data(data)

# Counts the number of events in each analysis bin, to give the total observed
# events in each bin
binned_data = np.array([len(sorted_data[data_bin]) for data_bin in data_bin_slices])

# Sets up the Weighter class, that manages all the weight calculations
#weight_maker = weighter.Weighter(sorted_mc)
if args.model == "nusiprop":
    # Initialize nuSIprop object for nuSIprop model
    # Get initial values for nuSIprop
    # Note: astro_gamma is used as the spectral index (si) for nuSIprop
    astro_gamma_idx = parameter_names.index("astro_gamma")
    si_val = params[astro_gamma_idx]  # Use astro_gamma as si
    astro_norm_idx = parameter_names.index("astro_norm")
    astro_norm_val = params[astro_norm_idx]
    Mphi_idx = parameter_names.index("Mphi")
    g_idx = parameter_names.index("g")
    mntot_idx = parameter_names.index("mntot")
    Mphi_val = params[Mphi_idx]  # Already in GeV
    g_val = params[g_idx]
    mntot_val = params[mntot_idx]
    # norm_base is used internally by nuSIprop, astro_norm scales it later
    #norm_base = 1e-18
    print('before initializing nuSIprop object')
    print(Mphi_val, g_val, mntot_val, si_val, astro_norm_val, si_val)
    
    # Initialize nuSIprop object with initial parameter values
    # The set_parameters() method will be called during the fit to update values
    # nuSIprop.pyprop expects mphi in eV
    # nuSIprop looks for xsec files relative to current working directory,
    # so we need to change to nuSIprop directory temporarily
    gc.collect()
    original_cwd = os.getcwd()
    print('original_cwd', original_cwd)
    try:
        os.chdir(nuSIprop_path)
        print('chdir to nuSIprop_path', nuSIprop_path)
        nuSIprop_obj = nuSIprop.pyprop(
            mphi=Mphi_val*1e6, g=g_val, si=si_val, norm=1e-18, mntot=mntot_val,
            majorana=True, non_resonant=True, normal_ordering=True,
            N_bins_E=200, lEmin=13-0.1, lEmax=16.01, zmax=4, flav=2, phiphi=True
        )
        print('initialized nuSIprop object')
    finally:
        os.chdir(original_cwd)
    weight_maker = weighter_original.Weighter(sorted_mc, model=args.model, nuSIprop=nuSIprop_obj)
else:
    weight_maker = weighter_original.Weighter(sorted_mc, model=args.model)

# For profile likelihood: exclude priors on fixed parameters
# This ensures we compute max_η [L(θ_fixed, η) * π(η)] without π(θ_fixed)
exclude_prior_indices = []
for i, (name, fixed) in enumerate(zip(parameter_names, is_fixed)):
    if fixed:
        exclude_prior_indices.append(i)

# A wrapper function that handles fits with fixed parameters
def calcLLH_fitted_func(is_fitted, params, exclude_prior_indices):
    # Track evaluation count and last LLH for progress printing
    eval_count = [0]  # Use list to allow modification in nested function
    last_llh = [None]  # Track last LLH to detect changes
    
    def func(
        fitted_params,
        parameter_names,
        priors,
        mc_bin_slices,
        binned_data,
        weights,
        livetime,
    ):
        params[:][is_fitted] = fitted_params
        llh, grads = likelihood.calcLLH(
            params,
            parameter_names,
            priors,
            mc_bin_slices,
            binned_data,
            weights,
            livetime,
            exclude_prior_indices=exclude_prior_indices,
        )
        
        # Print progress every 5 evaluations or when LLH changes significantly
        eval_count[0] += 1
        print_this = False
        if eval_count[0] % 5 == 0:
            print_this = True
        elif last_llh[0] is not None and abs(llh - last_llh[0]) > 0.1:
            print_this = True
        
        if print_this:
            print(f"  Eval {eval_count[0]}: -LLH = {llh:.6f}")
        
        last_llh[0] = llh
        return llh, np.array(grads[0])[is_fitted]

    return func


calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params), exclude_prior_indices)

bounds_list = []
fitted_params_list = []
llh_list = []
info_list = []

# It has been observed that the log likelihood space is bimodal as a function
# of the DOM efficiency. To account for this, we split the allowed boundaries
# of the DOM efficiency in the fit, and separately fit for both sets of
# boundaries.
if args.fix_epsilon_dom:
    # If the DOM efficiency paramter is fixed, don't split the boundaries
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    bounds_list.append(bounds)
else:
    # If the DOM efficiency is fitted, split the allowed boundaries in DOM
    # Efficiency space and create two sets of boundaries.
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    index = parameter_names.index("epsilon_dom")
    bounds_low = np.copy(bounds)
    bounds_high = np.copy(bounds)
    bounds_low[index] = [0.8, 0.99]
    bounds_high[index] = [0.99, 1.25]

    bounds_list.append(bounds_low)
    bounds_list.append(bounds_high)

start = time.time()
print("Running fit")

for idx, bounds in enumerate(bounds_list):
    # Print which interval we're fitting
    if not args.fix_epsilon_dom and len(bounds_list) == 2:
        interval_name = "low [0.8, 0.99]" if idx == 0 else "high [0.99, 1.25]"
        print(f"Fitting epsilon_DOM {interval_name} interval...")
    # Function that runs the fit.
    fitted_params, llh, info = fmin_l_bfgs_b(
        calcLLH,
        x0=params[is_fitted],
        args=(
            parameter_names,
            priors,
            mc_bin_slices,
            binned_data,
            weight_maker,
            livetime,
        ),
        bounds=bounds[is_fitted],
        m=10,
        pgtol=1e-6 if args.model == "nusiprop" else 1e-15,  # Relaxed for nuSIprop (finite-diff gradients)
        factr=1e7 if args.model == "nusiprop" else 1e4,      # Relaxed for nuSIprop
    )

    fitted_params_list.append(fitted_params)
    llh_list.append(llh)
    info_list.append(info)
    
    # Print progress after each fit
    if not args.fix_epsilon_dom and len(bounds_list) == 2:
        interval_name = "low [0.8, 0.99]" if idx == 0 else "high [0.99, 1.25]"
        print(f"  Completed {interval_name} interval: -LLH = {llh:.6f}")

# Pick out the information from the fit with the lowest log likelihood.
min_index = np.argmin(llh_list)
BF_fitted_params = fitted_params_list[min_index]
BF_llh = llh_list[min_index]
BF_info = info_list[min_index]

end = time.time()

print("Fit took " + str(end - start) + " seconds")
BF_params = params[:]
BF_params[is_fitted] = BF_fitted_params

print("Best Fit -LLH: ", BF_llh)
print("Best Fit Paramters:")
for param, BF_param in zip(parameter_names, BF_params):
    print("\t{}: \t{}".format(param, BF_param))

print(BF_info)
print(BF_llh)
print(BF_params.tolist())

# Output both fit results for epsilon_DOM bi-modality
if not args.fix_epsilon_dom and len(llh_list) == 2:
    print("\n=== Both epsilon_DOM interval fits ===")
    for i, (llh, fitted_params, info) in enumerate(zip(llh_list, fitted_params_list, info_list)):
        interval_name = "low" if i == 0 else "high"
        epsilon_range = "[0.8, 0.99]" if i == 0 else "[0.99, 1.25]"
        print(f"\nFit {i+1} (epsilon_DOM {interval_name} interval, {epsilon_range}):")
        print(f"  -LLH: {llh}")
        fit_params = params[:]
        fit_params[is_fitted] = fitted_params
        print("  Parameters:")
        for param, fit_param in zip(parameter_names, fit_params):
            print("\t{}: \t{}".format(param, fit_param))
        print(f"  Interval: {interval_name}")
        print(f"  Epsilon_DOM range: {epsilon_range}")

print('LLH list: ', llh_list)
print('Fitted params list: ', fitted_params_list)
print('Info list: ', info_list)