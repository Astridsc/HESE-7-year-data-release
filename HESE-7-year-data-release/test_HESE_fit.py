"""
This file runs a fit of the data over a single power-law astrophysical flux
model. The initial value of the fits can be modified within the file or through
command line arguments. One can also choose to fix certain parameters in the
fit, where the fixed value is kept at the initial value
"""

"""
OUTPUT FROM FITTING SPL:
(nusiprop) astridaurora@LAPTOP-H6QI3I8N:~/HESE-7-year-data-release/HESE-7-year-data-release$ python HESE_fit.py
Running fit
Fit took 1298.063502073288 seconds
Best Fit -LLH:  122.95919802083874
Best Fit Paramters:
        cr_delta_gamma:         -0.05309198828568753
        nunubar_ratio:  0.998164256210702
        anisotropy_scale:       1.0007247919049886
        astro_gamma:    2.8737764773857943
        astro_norm:     6.365300091182592
        conv_norm:      1.006210702376819
        epsilon_dom:    0.9519225902130987
        epsilon_head_on:        -0.05499094382686424
        muon_norm:      1.1868488857218278
        kpi_ratio:      1.0001423496123587
        prompt_norm:    0.0
{'grad': array([-1.19226273e-04,  1.89395003e-04,  5.31073096e-04, -3.95635879e-04,
       -4.64323521e-05,  1.76296805e-04,  2.36586607e-03,  9.81904293e-04,
       -7.87427092e-04,  3.45024198e-04,  1.89237667e-02]), 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'funcalls': 90, 'nit': 59, 'warnflag': 0}
122.95919802083874
[-0.05309198828568753, 0.998164256210702, 1.0007247919049886, 2.8737764773857943, 6.365300091182592, 1.006210702376819, 0.9519225902130987, -0.05499094382686424, 1.1868488857218278, 1.0001423496123587, 0.0]


LP MODEL OUTPUT:
(nusiprop) astridaurora@LAPTOP-H6QI3I8N:~/HESE-7-year-data-release/HESE-7-year-data-release$ python HESE_fit.py
Running fit
Fit took 1542.5534834861755 seconds
Best Fit -LLH:  122.92789302853468
Best Fit Paramters:
        cr_delta_gamma:         -0.05282593018858051
        nunubar_ratio:  0.9981462730693877
        anisotropy_scale:       1.000723878214125
        astro_gamma:    2.7762273797524797
        astro_norm:     6.265585095843488
        conv_norm:      1.0036685935384948
        epsilon_dom:    0.9520803131571901
        epsilon_head_on:        -0.05568771019336483
        muon_norm:      1.1895043671868821
        kpi_ratio:      1.0000262733582987
        prompt_norm:    0.0
        beta:   0.09082969303229133
{'grad': array([-0.00154187,  0.00049199,  0.00097954, -0.0009458 , -0.00057086,
       -0.00237775, -0.0123779 ,  0.00099125, -0.00055583, -0.00104512,
        0.02623515, -0.00058961]), 'task': 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH', 'funcalls': 85, 'nit': 58, 'warnflag': 0}
122.92789302853468
[-0.05282593018858051, 0.9981462730693877, 1.000723878214125, 2.7762273797524797, 6.265585095843488, 1.0036685935384948, 0.9520803131571901, -0.05568771019336483, 1.1895043671868821, 1.0000262733582987, 0.0, 0.09082969303229133]"""



import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import argparse
import time

import weighter
import binning
import data_loader
import autodiff
import likelihood
import det_sys_weights

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
parser.add_argument("--beta", default=0.0, type=float,
                    help="set initial log-parabola curvature")
parser.add_argument("--cutoff_energy", default=0.0, type=float,
                    help="set initial cutoff energy in GeV (0 means no cutoff)")
parser.add_argument("--Mphi", default=5.0, type=float,
                    help="set initial Mphi parameter for nuSIprop (in GeV)")
parser.add_argument("--g", default=0.1, type=float,
                    help="set initial g parameter for nuSIprop")
parser.add_argument("--mntot", default=0.1, type=float,
                    help="set initial mntot parameter for nuSIprop")
parser.add_argument("--model", default="spl", type=str,
                    choices=["spl", "lp", "cutoff", "nusiprop"],
                    help="astrophysical flux model: 'spl' (single power law), 'lp' (log-parabola), 'cutoff' (exponential cutoff), 'nusiprop' (nuSIprop)")


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
parser.add_argument("--fix_beta", action="store_true",
                    help="fix log-parabola curvature in fit")
parser.add_argument("--fix_cutoff_energy", action="store_true",
                    help="fix cutoff energy in fit")
parser.add_argument("--fix_Mphi", action="store_true",
                    help="fix Mphi parameter in fit (nuSIprop)")
parser.add_argument("--fix_g", action="store_true",
                    help="fix g parameter in fit (nuSIprop)")
parser.add_argument("--fix_mntot", action="store_true",
                    help="fix mntot parameter in fit (nuSIprop)")

args = parser.parse_args()

livetime = 227708167.68

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
    args.beta,
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
    (-0.05, 0.05, -np.inf, np.inf),
    (1.0, 0.1, 0.0, 2.0),
    (1.0, 0.2, 0.0, 2.0),
    (None, None, -np.inf, np.inf),
    (None, None, 0.0, np.inf),
    (1.0, 0.4, 0.0, np.inf),
    (0.99, 0.1, 0.8, 1.25),
    (0.0, 0.5, -3.82, 2.18),
    (1.0, 0.5, 0.0, np.inf),
    (1.0, 0.1, 0.0, np.inf),
    (None, None, 0.0, np.inf),
    (None, None, -np.inf, np.inf),
    (None, None, 1e4, 1e7, "log_uniform"),  # cutoff_energy: log-uniform prior, bounds in GeV
]

# Add nuSIprop priors if using nuSIprop model
if args.model == "nusiprop":
    priors.extend([
        (None, None, 0.03, 100, "log_uniform"),  # Mphi: log-uniform in GeV
        (None, None, 1e-4, 1.0, "log_uniform"),      # g: log-uniform
        (None, None, 0.06, 0.15),              # mntot: uniform
    ])

# Check that all initial parameters are within prior bounds
# Note: We'll check this after setting fixed parameters, so skip check for now
# The check will be done after we set fixed values for model-specific parameters

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
    args.fix_beta,
    args.fix_cutoff_energy,
]

# Add nuSIprop fix flags if using nuSIprop model
if args.model == "nusiprop":
    is_fixed.extend([args.fix_Mphi, args.fix_g, args.fix_mntot])

# Automatically fix irrelevant parameters based on model (unless explicitly set)
# This ensures all model-specific parameters are fitted while irrelevant ones are fixed
if not args.fix_beta and not args.fix_cutoff_energy:
    if args.model == "spl":
        # SPL: fix beta and cutoff_energy (not used)
        beta_idx = parameter_names.index("beta")
        cutoff_idx = parameter_names.index("cutoff_energy")
        is_fixed[beta_idx] = True
        is_fixed[cutoff_idx] = True
        params[beta_idx] = 0.0
        # Set cutoff_energy to lower bound of prior (1e4 GeV) when fixed
        params[cutoff_idx] = priors[cutoff_idx][2]  # Use lower bound from prior
    elif args.model == "lp":
        # LP: fix cutoff_energy (not used), but fit beta
        cutoff_idx = parameter_names.index("cutoff_energy")
        is_fixed[cutoff_idx] = True
        # Set cutoff_energy to lower bound of prior (1e4 GeV) when fixed
        params[cutoff_idx] = priors[cutoff_idx][2]  # Use lower bound from prior
    elif args.model == "cutoff":
        # Cutoff: fix beta (not used), but fit cutoff_energy
        beta_idx = parameter_names.index("beta")
        is_fixed[beta_idx] = True
        params[beta_idx] = 0.0
    elif args.model == "nusiprop":
        # nuSIprop: fix beta=0 and cutoff_energy (not used)
        # For cutoff_energy, set to lower bound of prior since it's not used
        beta_idx = parameter_names.index("beta")
        cutoff_idx = parameter_names.index("cutoff_energy")
        is_fixed[beta_idx] = True
        is_fixed[cutoff_idx] = True
        params[beta_idx] = 0.0
        # Set cutoff_energy to lower bound of prior (1e4 GeV) when fixed
        params[cutoff_idx] = priors[cutoff_idx][2]  # Use lower bound from prior

# Check that all initial parameters are within prior bounds (after setting fixed values)
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

# bin_data takes an MC/data numpy array as input, and returns
# 0: the events rearranged such that events are grouped by analysis bins.
# 1: the list of bin slices for each analysis bin.
sorted_mc, mc_bin_slices = binning.bin_data(mc)
sorted_data, data_bin_slices = binning.bin_data(data)

# Counts the number of events in each analysis bin, to give the total observed
# events in each bin
binned_data = np.array([len(sorted_data[data_bin]) for data_bin in data_bin_slices])

# Sets up the Weighter class, that manages all the weight calculations
# Pass model type to weighter
if args.model == "nusiprop":
    # Initialize nuSIprop object for nuSIprop model
    import nuSIprop
    # Get initial values for nuSIprop
    # Note: astro_gamma is used as the spectral index (si) for nuSIprop
    astro_gamma_idx = parameter_names.index("astro_gamma")
    si_val = params[astro_gamma_idx]  # Use astro_gamma as si
    Mphi_idx = parameter_names.index("Mphi")
    g_idx = parameter_names.index("g")
    mntot_idx = parameter_names.index("mntot")
    Mphi_val = params[Mphi_idx]  # Already in GeV
    g_val = params[g_idx]
    mntot_val = params[mntot_idx]
    # norm_base is used internally by nuSIprop, astro_norm scales it later
    norm_base = 1e-18
    
    # Initialize nuSIprop object with initial parameter values
    # The set_parameters() method will be called during the fit to update values
    # nuSIprop.pyprop expects mphi in GeV
    nuSIprop_obj = nuSIprop.pyprop(
        mphi=Mphi_val*1e6, g=g_val, si=si_val, norm=norm_base, mntot=mntot_val,
        majorana=True, non_resonant=True, normal_ordering=True,
        N_bins_E=300, lEmin=13, lEmax=16.01, zmax=5, flav=2, phiphi=False
    )
    weight_maker = weighter.Weighter(sorted_mc, nuSIprop=nuSIprop_obj, model=args.model)
else:
    weight_maker = weighter.Weighter(sorted_mc, nuSIprop=False, model=args.model)

# A wrapper function that handles fits with fixed parameters
def calcLLH_fitted_func(is_fitted, params):
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
        )
        return llh, np.array(grads[0])[is_fitted]

    return func


calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params))

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

for bounds in bounds_list:
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
        pgtol=1e-18,
        factr=1e4,
    )
    print('LLH: ', llh)
    fitted_params_list.append(fitted_params)
    llh_list.append(llh)
    info_list.append(info)

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
