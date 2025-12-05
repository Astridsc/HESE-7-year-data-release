
import sys
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.special import factorial

import argparse
import time

import weighter
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

# Add command line arguments for parameters
parser.add_argument("--astro_norm", default=6.0, type=float, help="set initial astrophysical normalization")
parser.add_argument("--astro_gamma", default=2.5, type=float, help="set initial astrophysical spectral index")
parser.add_argument("--fix_astro_norm", action="store_true", help="fix astrophysical normalization in fit")
parser.add_argument("--fix_astro_gamma", action="store_true", help="fix astrophysical spectral index in fit")
parser.add_argument("--model", default="spl", type=str, choices=["spl", "cutoff", "nusiprop"],
                    help="astrophysical flux model")

# Add nuSIprop parameters if needed
parser.add_argument("--Mphi", default=5.0, type=float, help="set initial Mphi parameter for nuSIprop (in GeV)")
parser.add_argument("--g", default=0.1, type=float, help="set initial g parameter for nuSIprop")
parser.add_argument("--mntot", default=0.1, type=float, help="set initial mntot parameter for nuSIprop")
parser.add_argument("--fix_Mphi", action="store_true", help="fix Mphi parameter in fit (nuSIprop)")
parser.add_argument("--fix_g", action="store_true", help="fix g parameter in fit (nuSIprop)")
parser.add_argument("--fix_mntot", action="store_true", help="fix mntot parameter in fit (nuSIprop)")

args = parser.parse_args()
"""
energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]

norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16, 0.0]"""

energy_edges_MESE = [10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0]
energy_centers_MESE = [14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9]
norm_segmented = [4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25]
sigma_upper = [0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24]
sigma_lower = [0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16]

energy_edges_MESE = np.asarray(energy_edges_MESE)
energy_centers_MESE = np.asarray(energy_centers_MESE)
energy_edges_MESE *= 1e3 # GeV
energy_centers_MESE *= 1e3 # GeV

norm_segmented = np.asarray(norm_segmented)
sigma_upper = np.asarray(sigma_upper)
sigma_lower = np.asarray(sigma_lower)

norm_segmented *= 1e-8 # GeV^-1 * cm^-2 * s^-1 * sr^-1
sigma_upper *= 1e-8 # GeV^-1 * cm^-2 * s^-1 * sr^-1
sigma_lower *= 1e-8 # GeV^-1 * cm^-2 * s^-1 * sr^-1

livetime = 227708167.68  # HESE livetime

# Load effective area data
base_path = os.path.dirname(os.path.abspath(__file__))
effective_area_path = os.path.join(base_path, 'notebooks', 'effective_area_4_to_7.csv')
#effective_area_path = os.path.join(base_path, 'Astrid', 'effective_areas_dataframes', 'effective_areas_by_flavor_gen2.csv')
effective_area_df = pd.read_csv(effective_area_path)
# Rename the first (unnamed) column to 'energy_bins'
effective_area_df.rename(columns={effective_area_df.columns[0]: 'energy_bins'}, inplace=True)
effective_area_df['effective_area_m2'] = effective_area_df['nu_e'] + effective_area_df['nu_mu'] + effective_area_df['nu_tau']
#effective_area_df['effective_area_m2'] *= 0.1
 
def interpolate_effective_area(effective_area_df, energies_interpolate):
    effective_area = effective_area_df['effective_area_m2'].values
    energy = effective_area_df['energy_bins'].values
    
    # Clip energies to available range to avoid NaN from extrapolation
    energy_min = energy.min()
    energy_max = energy.max()
    energies_clipped = np.clip(energies_interpolate, energy_min, energy_max)
    
    # Interpolate (or use boundary values for clipped energies)
    result = interp1d(energy, effective_area, bounds_error=False, 
                      fill_value=(effective_area[0], effective_area[-1]))(energies_clipped)
    
    return result


def initialize_nuSIprop(args):
    evolver = nuSIprop.pyprop(mphi=args.Mphi*1e6, g=args.g, si=args.astro_gamma, norm=1e-18, mntot=args.mntot, 
                                    majorana=True, non_resonant=True, normal_ordering=True,
                                    N_bins_E=300, lEmin=13, lEmax=16.01, zmax=5, flav=2, phiphi=False)
    evolver.evolve()
    return evolver





def get_data_bins(segmented_flux, effective_area_df, livetime):
    # Get number of most likely astrophysical events in each energy bin
    N = np.zeros(len(energy_centers_MESE))
    for i in range(len(energy_edges_MESE)-1):
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i+1]
        E = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        Aeff_i = interpolate_effective_area(effective_area_df, E)
        integral_i = np.trapz(Aeff_i * E**(-2), E)
        N[i] = integral_i * livetime * segmented_flux[i]
        
    return N 


def get_data_uncertainty(sigma_upper, sigma_lower, effective_area_df, livetime):
    sigma_upp = np.zeros(len(energy_centers_MESE))
    sigma_low = np.zeros(len(energy_centers_MESE))
    for i in range(len(energy_edges_MESE)-1):
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i+1]
        E = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        Aeff_i = interpolate_effective_area(effective_area_df, E)
        integral_i = np.trapz(Aeff_i * E**(-2), E)
        sigma_upp[i] = integral_i * livetime * sigma_upper[i]
        sigma_low[i] = integral_i * livetime * sigma_lower[i]
    return sigma_upp, sigma_low


def get_model_bins(model_flux_function, params, effective_area_df, livetime):
    # Validate parameters at the start
    if np.any(~np.isfinite(params)):
        print(f"ERROR: Invalid parameters in get_model_bins: {params}")
        return np.zeros(len(energy_centers_MESE))
    
    N = np.zeros(len(energy_centers_MESE))
    for i in range(len(energy_edges_MESE)-1):
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i+1]
        E = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        Aeff_i = interpolate_effective_area(effective_area_df, E)
        
        # Debug: check for NaN in interpolation
        if np.any(np.isnan(Aeff_i)):
            print(f"WARNING: NaN in Aeff_i for bin {i}, Emin={Emin:.2e}, Emax={Emax:.2e}")
            print(f"  Energy range in effective_area_df: {effective_area_df['energy_bins'].min():.2e} to {effective_area_df['energy_bins'].max():.2e}")
        
        flux_i = model_flux_function(E, params)
        
        # Debug: check for NaN in flux
        if np.any(np.isnan(flux_i)):
            print(f"WARNING: NaN in flux_i for bin {i}")
            print(f"  params: {params}")
            print(f"  E range: {E.min():.2e} to {E.max():.2e}")
            print(f"  flux_i sample: {flux_i[:5]}")
        
        integrand = Aeff_i * flux_i
        """if np.any(np.isnan(integrand)):
            print(f"WARNING: NaN in integrand for bin {i}")
            print(f"  Aeff_i has NaN: {np.any(np.isnan(Aeff_i))}")
            print(f"  flux_i has NaN: {np.any(np.isnan(flux_i))}")"""
        
        integral_i = np.trapz(integrand, E)
        N[i] = integral_i * livetime 
    
    # Debug: print summary of what was computed
    # (commented out to reduce noise, uncomment if needed)
    # print(f"  get_model_bins: params={params}, N={N}")
    
    return N



def log_likelihood_poisson(N_data, N_model):
    #print("N_data: ", N_data)
    #print("N_model: ", N_model)
    if N_data == 0:
        #print('N_data is 0, N_model is: ', N_model)
        return -N_model
    else:
        return -N_model + N_data*np.log(N_model) - np.log(factorial(N_data))


def get_TS(N_data, N_model, sigma_upp, sigma_low):
    TS = np.zeros(len(energy_centers_MESE))
    
    for i in range(len(energy_centers_MESE)):
        if N_model[i] > N_data[i]:
            sigma = sigma_upp[i]
        else:
            sigma = - sigma_low[i]
            
        if energy_centers_MESE[i] < 1e5:   # 100TeV
            TS[i] = (N_data[i] - N_model[i])**2 / (sigma**2)  # Pearson's chi2
            if np.isclose(N_model[i], N_data[i], rtol=1e-10):
                print(f"    Low E bin {i}: TS = {TS[i]} (should be 0 when N_model==N_data)")
            
        else:
            """
            print('första termen, i:', i, log_likelihood_poisson(N_data[i], N_model[i]))
            print('andra termen: ', i, log_likelihood_poisson(N_data[i], N_data[i]))
            if log_likelihood_poisson(N_data[i], N_model[i]) - log_likelihood_poisson(N_data[i], N_data[i])  > 0:
                print('täljare positiv!')
            print('tredje termen: ', i, log_likelihood_poisson(N_data[i], N_data[i] + sigma ))
            print('fjärde termen, i:', i, log_likelihood_poisson(N_data[i], N_data[i]))"""
            #print('numerator i: ', i, (log_likelihood_poisson(N_data[i], N_model[i]) - log_likelihood_poisson(N_data[i], N_data[i])))
            #print("denominator i: ", i, (log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i])))
            #print('data, sigma:', i, N_data[i], sigma)
            TS[i] = -2* (log_likelihood_poisson(N_data[i], N_model[i]) - log_likelihood_poisson(N_data[i], N_data[i])) / (log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i]))
            #assert log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i]) < 0
    return TS


def model_flux_spl(E, params):
    """Single power law flux model: norm * (E/E_pivot)^(-gamma)"""
    astro_norm, astro_gamma = params
    E_pivot = 1e5  # GeV
    return astro_norm * (E / E_pivot)**(-astro_gamma) * 1e-18 / 6.0


def model_flux_cutoff(E, params):
    """Exponential cutoff flux model: norm * (E/E_pivot)^(-gamma) * exp(-E/E_cutoff)"""
    astro_norm, astro_gamma, cutoff_energy = params
    E_pivot = 1e5  # GeV
    power_law = (E / E_pivot)**(-astro_gamma)
    cutoff = np.exp(-E / cutoff_energy)
    return astro_norm * power_law * cutoff * 1e-18 / 6.0


# Load effective area data (you'll need to provide this)
# For now, creating a placeholder - you'll need to load your actual effective area data
# effective_area_df = load_effective_area_data()  # You need to implement this

# Compute data bins from segmented flux
# N_data = get_data_bins(norm_segmented, effective_area_df, livetime)

# Set up parameters
parameter_names = ["astro_norm", "astro_gamma"]
params = [args.astro_norm, args.astro_gamma]

# Add nuSIprop parameters if using nuSIprop model
if args.model == "nusiprop":
    parameter_names.extend(["Mphi", "g", "mntot"])
    params.extend([args.Mphi, args.g, args.mntot])

params = np.array(params)

# Set up priors (similar to HESE_fit.py)
priors = [
    (None, None, 0.5, 12.0),  # astro_norm: uniform prior (must be positive)
    (None, None, 1.0, 5.0),  # astro_gamma: uniform prior (reasonable range to prevent overflow)
]

if args.model == "nusiprop":
    priors.extend([
        (None, None, 0.03, 100, "log_uniform"),  # Mphi: log-uniform in GeV
        (None, None, 1e-4, 1.0, "log_uniform"),  # g: log-uniform
        (None, None, 0.06, 0.15),  # mntot: uniform
    ])

# Set up fixed parameters
is_fixed = [args.fix_astro_norm, args.fix_astro_gamma]

if args.model == "nusiprop":
    is_fixed.extend([args.fix_Mphi, args.fix_g, args.fix_mntot])

if np.any(is_fixed):
    print("Fixing parameters")
    for b, name, val in zip(is_fixed, parameter_names, params):
        if b:
            print(name + " = ", val)

is_fitted = [not b for b in is_fixed]

# A wrapper function that handles fits with fixed parameters (similar to HESE_fit.py)
def calcLLH_fitted_func(is_fitted, params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low):
    def func(fitted_params, parameter_names, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low):
        params[:][is_fitted] = fitted_params
        #print("params: ", params)
        ts, grads = calc_mese_ts(
            params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low
        )
        return ts, np.array(grads)[is_fitted]
    return func


def calc_mese_ts(params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, compute_grads=True):
    """
    Compute MESE test statistic (TS) for marginalization.
    
    This function uses your get_TS function to compute the test statistic.
    Uses finite differences to compute gradients.
    """
    # Compute model bins
    #print("params: ", params)
    N_model = get_model_bins(model_flux_function, params, effective_area_df, livetime)
    print("N_model: ", N_model)
    
    # Compute TS using your get_TS function
    TS_array = get_TS(N_data, N_model, sigma_upp, sigma_low)
    print("TS_array: ", TS_array)
    total_ts = np.sum(TS_array)  # Sum of TS values across all bins
    print("total_ts: ", total_ts)
    
    # Compute gradients using finite differences
    grads = np.zeros(len(params))
    if compute_grads:
        eps = 1e-3
        for i in range(len(params)):
            params_pert = params.copy()
            # Use relative perturbation for better numerical stability
            if params[i] != 0:
                params_pert[i] = params[i] * (1.0 + eps)
            else:
                params_pert[i] = params[i] + eps
            
            # Compute perturbed TS
            N_model_pert = get_model_bins(model_flux_function, params_pert, effective_area_df, livetime)
            TS_array_pert = get_TS(N_data, N_model_pert, sigma_upp, sigma_low)
            total_ts_pert = np.sum(TS_array_pert)
            
            # Compute gradient
            if params[i] != 0:
                grads[i] = (total_ts_pert - total_ts) / (params[i] * eps)
            else:
                grads[i] = (total_ts_pert - total_ts) / eps
    
    return total_ts, grads




# Select model flux function
if args.model == "spl":
    model_flux_function = model_flux_spl
elif args.model == "cutoff":
    # You'll need to add cutoff_energy to parameters if using cutoff model
    raise NotImplementedError("Cutoff model not yet implemented in MESE_fit")
elif args.model == "nusiprop":
    # You'll need to implement nuSIprop flux function for MESE
    raise NotImplementedError("nuSIprop model not yet implemented in MESE_fit")
else:
    raise ValueError(f"Unknown model: {args.model}")

# Compute N_data from segmented flux
N_data = get_data_bins(norm_segmented, effective_area_df, livetime)
print("N_data: ", N_data)
sigma_upp, sigma_low = get_data_uncertainty(sigma_upper, sigma_lower, effective_area_df, livetime)
print("sigma_upp: ", sigma_upp)
print("sigma_low: ", sigma_low)
# Run the fit
calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params),
                              effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low)

# Set bounds (you may want to adjust these based on your parameter ranges)
bounds = np.array([(prior[2], prior[3]) for prior in priors])

start = time.time()
print("Running fit")

print("params[is_fitted]: ", params[is_fitted])

fitted_params, ts, info = fmin_l_bfgs_b(
    calcLLH,
    x0=params[is_fitted],
    args=(parameter_names, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low),
    bounds=bounds[is_fitted],
    m=10,
    pgtol=1e-18,
    factr=1e4,
)

end = time.time()

print("Fit took " + str(end - start) + " seconds")
BF_params = params[:]
BF_params[is_fitted] = fitted_params

print("Best Fit TS: ", ts)
print("Best Fit Parameters:")
for param, BF_param in zip(parameter_names, BF_params):
    print("\t{}: \t{}".format(param, BF_param))

print(info)
print(ts)
print(BF_params.tolist())

