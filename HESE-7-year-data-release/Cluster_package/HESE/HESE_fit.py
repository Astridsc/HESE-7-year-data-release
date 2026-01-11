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
import json
import os
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
# nuSIprop import will be handled after argument parsing to determine correct path


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
    default=0.0,
    type=float,
    help="set initial atmospheric prompt neutrino flux normalization",
)

parser.add_argument("--model_type", default="nusiprop", type=str,
                    choices=["nusiprop", "regular"],
                    help="model implemented with secret interactions: nusiprop, or without secret interactions (regular)")
"""parser.add_argument(
    "--model", default="spl", type=str,
                    choices=["spl", "cutoff", "nusiprop"],
                    help="astrophysical flux model: 'spl' (single power law), 'cutoff' (exponential cutoff), 'nusiprop' (nuSIprop)")"""
parser.add_argument("--model", default="spl", type=str,
                    choices=["spl", "bpl", "lp", "cutoff"],
                    help="astrophysical flux model: 'spl' (single power law), 'bpl' (broken power law), 'lp' (log-parabola), 'cutoff' (exponential cutoff)")

parser.add_argument("--Mphi", default=1.0, type=float,
                    help="set initial Mphi parameter for nuSIprop (in GeV)")
parser.add_argument("--g", default=0.01, type=float,
                    help="set initial g parameter for nuSIprop")
parser.add_argument("--mntot", default=0.1, type=float,
                    help="set initial mntot parameter for nuSIprop")
parser.add_argument("--si2", default=2.0, type=float,
                    help="set initial spectral index for broken power law or log-parabola")
parser.add_argument("--E_break", default=5e4, type=float,
                    help="set initial break energy for broken power law (in GeV)")
parser.add_argument("--cutoff_energy", default=1e5, type=float,
                    help="set initial cutoff energy parameter for exponential cutoff")

def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument("--majorana", type=str_to_bool, default=True,
                    help="use Majorana (True) or Dirac (False) neutrinos for nuSIprop (default: True)")
parser.add_argument("--normal", type=str_to_bool, default=True,
                    help="use normal (True) or inverted (False) mass ordering for nuSIprop (default: True)")



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
parser.add_argument("--fix_cutoff_energy", action="store_true",
                    help="fix cutoff energy parameter for exponential cutoff in fit")
parser.add_argument("--fix_si2", action="store_true",
                    help="fix spectral index for broken power law or log-parabola in fit")
parser.add_argument("--fix_E_break", action="store_true",
                    help="fix break energy for broken power law in fit")
parser.add_argument("--pgtol", type=float, default=None,
                    help="gradient tolerance for L-BFGS-B optimizer (default: 1e-10 for nusiprop, 1e-15 otherwise)")
parser.add_argument("--factr", type=float, default=None,
                    help="convergence factor for L-BFGS-B optimizer (default: 1e7 for nusiprop, 1e4 otherwise)")
parser.add_argument("--m", type=int, default=None,
                    help="number of corrections used in L-BFGS-B (default: 20)")
parser.add_argument("--maxiter", type=int, default=None,
                    help="maximum number of iterations for L-BFGS-B (default: 500)")
parser.add_argument("--cluster_mode", action="store_true", default=False,
                   help="run in cluster mode and save results to JSONL file")
parser.add_argument("--output_dir", type=str, default=None,
                   help="output directory for cluster mode results (required if --cluster_mode)")
parser.add_argument("--nuSI", type=str_to_bool, default=True,
                   help="Enable nuSIprop secret interactions (default: True). If False, fixes Mphi, g, mntot and sets g=1e-30")
parser.add_argument("--HESE12", type=str_to_bool, default=False,
                   help="Use HESE12 data instead of HESE data (default: False)")

args = parser.parse_args()

# Add nuSIprop to path - path depends on cluster_mode flag
base_path = os.path.dirname(os.path.abspath(__file__))
if args.cluster_mode:
    # Cluster path (../../nuSIprop-main-new from this file's location)
    nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop-main-new'))
    print('cluster mode')
else:
    # Local/standalone path (../../../../nuSIprop-main-new from this file's location)
    nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', '..', '..', 'nuSIprop-main-new'))

if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
print('nuSIprop_path', nuSIprop_path)
import nuSIprop


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
]

# Add nuSIprop parameters if using nuSIprop model
if args.model_type == "nusiprop":
    parameter_names.extend(["Mphi", "g", "mntot"])
    
if args.model == "bpl":
    parameter_names.extend(["si2", "E_break"])
elif args.model == "lp":
    parameter_names.extend(["si2"])
elif args.model == "cutoff":
    parameter_names.extend(["cutoff_energy"])
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
]

# Add nuSIprop parameters if using nuSIprop model
if args.model_type == "nusiprop":
    # If --nuSI False, override g to 1e-30
    g_value = 1e-30 if not args.nuSI else args.g
    params.extend([args.Mphi, g_value, args.mntot])  # Mphi in GeV
    
if args.model == "bpl":
    params.extend([args.si2, args.E_break])
elif args.model == "lp":
    params.extend([args.si2])
elif args.model == "cutoff":
    params.extend([args.cutoff_energy])
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
    (None, None, 1.5, 3.5), # astro_gamma, Uniform prior
    #(None, None, -np.inf, np.inf), # astro_gamma, Uniform prior
    (None, None, 0.0, np.inf), # astro_norm, Uniform prior
    (1.0, 0.4, 0.0, np.inf), # conv_norm, Gaussian prior
    (0.99, 0.1, 0.8, 1.25), # epsilon_dom, Gaussian prior
    (0.0, 0.5, -3.82, 2.18), # epsilon_head_on, Gaussian prior
    (1.0, 0.5, 0.0, np.inf), # muon_norm, Gaussian prior
    (1.0, 0.1, 0.0, np.inf), # kpi_ratio, Gaussian prior
    (None, None, 0.0, 4.0), # prompt_norm, Uniform prior
    #(None, None, -np.inf, np.inf), # prompt_norm, Uniform prior
]

# Add nuSIprop priors if using nuSIprop model
if args.model_type == "nusiprop":
    # Set g prior based on --nuSI flag
    if args.nuSI:
        g_prior = (None, None, 1e-4, 1, "log_uniform")  # g: log-uniform with wider range
    else:
        g_prior = (None, None, 1e-31, 1, "log_uniform")  # g: log-uniform with tighter range for no-SI case
    
    priors.extend([
        (None, None, 0.02, 100+10, "log_uniform"),  # Mphi: log-uniform in GeV
        g_prior,  # g: log-uniform (range depends on --nuSI)
        (None, None, 0.06, 0.15),                # mntot: uniform
    ])
    
if args.model == "bpl":
    priors[3] = (None, None, 0.0, 3.5) # si1 / astro_gamma, Uniform prior. Change for bpl to match 'prior' from MESE paper?
    priors.extend([
        (None, None, 2.0, 3.5), # si2: uniform prior
        (None, None, 1e4, 6e5), # E_break: log-uniform prior
    ])
elif args.model == "lp":
    priors.extend([
        (None, None, 1e-3, 2.0), # si2: uniform prior
    ])
elif args.model == "cutoff":
    priors.extend([
        (None, None, 1e5, 1e7, "log_uniform"), # cutoff_energy, log-uniform prior
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
]

# Add nuSIprop fix flags if using nuSIprop model
if args.model_type == "nusiprop":
    # If --nuSI False, automatically fix Mphi, g, and mntot
    if not args.nuSI:
        is_fixed.extend([True, True, True])  # Fix Mphi, g, mntot
    else:
        is_fixed.extend([args.fix_Mphi, args.fix_g, args.fix_mntot])
    
if args.model == "bpl":
    is_fixed.extend([args.fix_si2, args.fix_E_break])
elif args.model == "lp":
    is_fixed.extend([args.fix_si2])
elif args.model == "cutoff":
    is_fixed.extend([args.fix_cutoff_energy])
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
if args.HESE12:
    data = data_loader.load_data("./resources/data/HESE12_data.json")
    livetime = 12 * 365 * 24 * 3600
else:
    data = data_loader.load_data("./resources/data/HESE_data.json")
    livetime = 227708167.68

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
if args.model_type == "nusiprop":
    # Initialize nuSIprop object for nuSIprop model
    # Get initial values for nuSIprop
    # Note: astro_gamma is used as the spectral index (si) for nuSIprop
    astro_gamma_idx = parameter_names.index("astro_gamma")
    si_val = params[astro_gamma_idx]  # Use astro_gamma as si
    Mphi_idx = parameter_names.index("Mphi")
    g_idx = parameter_names.index("g")
    mntot_idx = parameter_names.index("mntot")
    Mphi_val = params[Mphi_idx]  # Already in GeV
    # If --nuSI False, use 1e-30 for g (already set in params, but ensure consistency)
    g_val = 1e-30 if not args.nuSI else params[g_idx]
    mntot_val = params[mntot_idx]
    if args.model == "bpl":
        si2_idx = parameter_names.index("si2")
        si2_val = params[si2_idx]
        E_break_idx = parameter_names.index("E_break")
        E_break_val = params[E_break_idx]
    elif args.model == "lp":
        si2_idx = parameter_names.index("si2")
        si2_val = params[si2_idx]
    elif args.model == "cutoff":
        cutoff_energy_idx = parameter_names.index("cutoff_energy")
        cutoff_energy_val = params[cutoff_energy_idx]

    # norm_base is used internally by nuSIprop, astro_norm scales it later
    norm_base = 1e-18
    
    # Initialize nuSIprop object with initial parameter values
    # The set_parameters() method will be called during the fit to update values
    # nuSIprop.pyprop expects mphi in eV
    # nuSIprop looks for xsec files relative to current working directory,
    # so we need to change to nuSIprop directory temporarily
    if args.model == "bpl":
        flux_model = 1
    elif args.model == "lp":
        flux_model = 2
    elif args.model == "cutoff":
        flux_model = 3
    else:
        flux_model = 0
    gc.collect()
    original_cwd = os.getcwd()
    try:
        os.chdir(nuSIprop_path)
        nuSIprop_obj = nuSIprop.pyprop(
            mphi=Mphi_val*1e6, g=g_val, si=si_val, norm=norm_base, mntot=mntot_val,
            majorana=args.majorana, non_resonant=True, normal_ordering=args.normal,
            N_bins_E=200, lEmin=13-0.1, lEmax=16.01, zmax=5, flav=2, phiphi=True,
            flux_model=flux_model,
            si2=si2_val if args.model in ["bpl", "lp"] else 2.5,
            E_break=E_break_val if args.model == "bpl" else 1e5,
        )
    finally:
        os.chdir(original_cwd)
    weight_maker = weighter_original.Weighter(sorted_mc, model=args.model, nuSIprop=nuSIprop_obj)
    gc.collect()
else:
    weight_maker = weighter_original.Weighter(sorted_mc, model=args.model)
    gc.collect()
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
        #print('llh', llh)
        print('grads', grads)
        print('params', params)
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
    if idx == 1:
        llh_list.append(np.inf)
        fitted_params_list.append(params[is_fitted])
        info_list.append(None)
        continue  # Skip high epsilon_dom interval (redundant)
    # Print which interval we're fitting
    if not args.fix_epsilon_dom and len(bounds_list) == 2:
        interval_name = "low [0.8, 0.99]" if idx == 0 else "high [0.99, 1.25]"
        print(f"Fitting epsilon_DOM {interval_name} interval...")
    # Function that runs the fit.
    # Use provided values or defaults based on model
    pgtol = args.pgtol if args.pgtol is not None else (1e-6 if args.model == "nusiprop" else 1e-15)
    factr = args.factr if args.factr is not None else (10 if args.model == "nusiprop" else 1e4)
    m = args.m if args.m is not None else 10
    maxiter = args.maxiter if args.maxiter is not None else 500
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
        m=m,  # Can be adjusted via command line
        pgtol=pgtol,  # Can be adjusted via command line
        factr=factr,  # Can be adjusted via command line
        maxiter=maxiter,  # Can be adjusted via command line
    )

    fitted_params_list.append(fitted_params)
    # Convert nan to inf so that argmin will select the finite value if one exists
    if np.isnan(llh):
        llh = np.inf
    llh_list.append(llh)
    info_list.append(info)
    
    # Print progress after each fit
    if not args.fix_epsilon_dom and len(bounds_list) == 2:
        interval_name = "low [0.8, 0.99]" if idx == 0 else "high [0.99, 1.25]"
        if np.isinf(llh):
            print(f"  Completed {interval_name} interval: -LLH = inf (fit failed)")
        else:
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

if np.isinf(BF_llh):
    print("Best Fit -LLH: inf (all fits failed)")
else:
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
        if np.isinf(llh):
            print(f"  -LLH: inf (fit failed)")
        else:
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

# Save results in cluster mode
if args.cluster_mode:
    if args.output_dir is None:
        print("ERROR: --output_dir is required when using --cluster_mode")
        sys.exit(1)
    
    # Make output directory path absolute
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "results.jsonl")
    results_file_abs = os.path.abspath(results_file)
    
    try:
        # Handle np.inf for JSON (JSON doesn't support infinity, use string)
        llh_json = float(BF_llh) if (BF_llh is not None and not np.isinf(BF_llh)) else "inf"
        
        # Convert parameters to dictionary
        params_dict = {}
        for param_name, param_value in zip(parameter_names, BF_params):
            params_dict[param_name] = float(param_value)
        
        # Create result dictionary
        result = {
            "llh": llh_json,
            "params": params_dict,
            "fit_time": float(end - start),
            "model": args.model,
        }
        
        # Add optimization parameters if they were specified
        if args.pgtol is not None:
            result["pgtol"] = args.pgtol
        if args.factr is not None:
            result["factr"] = args.factr
        if args.m is not None:
            result["m"] = args.m
        if args.maxiter is not None:
            result["maxiter"] = args.maxiter
        
        # Append to JSONL file (one JSON object per line, safe for parallel writes)
        with open(results_file_abs, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        print(f"\nResult saved to {results_file_abs}")
    except Exception as e:
        print(f"ERROR: Failed to save result to {args.output_dir}: {e}")
        print(f"Current working directory: {os.getcwd()}")
        raise