
import sys
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.special import factorial

import argparse
import time
import gc
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
nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop-main-new'))
if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
import nuSIprop

parser = argparse.ArgumentParser()

# Add command line arguments for parameters
parser.add_argument("--astro_norm", default=2.0, type=float, help="set initial astrophysical normalization")
parser.add_argument("--astro_gamma", default=2.5, type=float, help="set initial astrophysical spectral index")
parser.add_argument("--fix_astro_norm", action="store_true", help="fix astrophysical normalization in fit")
parser.add_argument("--fix_astro_gamma", action="store_true", help="fix astrophysical spectral index in fit")
parser.add_argument("--model", default="spl", type=str, choices=["spl", "cutoff", "nusiprop", "bpl", "lp"],
                    help="astrophysical flux model")

# Add cutoff model parameters
parser.add_argument("--cutoff_energy", default=1e5, type=float,
                    help="set initial cutoff energy parameter for exponential cutoff (in GeV)")
parser.add_argument("--fix_cutoff_energy", action="store_true",
                    help="fix cutoff energy parameter for exponential cutoff in fit")

# Add nuSIprop parameters if needed
parser.add_argument("--Mphi", default=5.0, type=float, help="set initial Mphi parameter for nuSIprop (in GeV)")
parser.add_argument("--g", default=0.1, type=float, help="set initial g parameter for nuSIprop")
parser.add_argument("--mntot", default=0.1, type=float, help="set initial mntot parameter for nuSIprop")
parser.add_argument("--fix_Mphi", action="store_true", help="fix Mphi parameter in fit (nuSIprop)")
parser.add_argument("--fix_g", action="store_true", help="fix g parameter in fit (nuSIprop)")
parser.add_argument("--fix_mntot", action="store_true", help="fix mntot parameter in fit (nuSIprop)")

# Add broken power law (bpl) model parameters
parser.add_argument("--astro_gamma1", default=2.0, type=float, help="set initial gamma1 for broken power law")
parser.add_argument("--astro_gamma2", default=3.0, type=float, help="set initial gamma2 for broken power law")
parser.add_argument("--E_break", default=5e4, type=float, help="set initial break energy for broken power law (in GeV)")
parser.add_argument("--fix_astro_gamma1", action="store_true", help="fix gamma1 parameter in fit (bpl)")
parser.add_argument("--fix_astro_gamma2", action="store_true", help="fix gamma2 parameter in fit (bpl)")
parser.add_argument("--fix_E_break", action="store_true", help="fix break energy parameter in fit (bpl)")

# Add log parabola (lp) model parameters
parser.add_argument("--alpha", default=2.5, type=float, help="set initial alpha for log parabola")
parser.add_argument("--beta", default=0.3, type=float, help="set initial beta for log parabola")
parser.add_argument("--fix_alpha", action="store_true", help="fix alpha parameter in fit (lp)")
parser.add_argument("--fix_beta", action="store_true", help="fix beta parameter in fit (lp)")

args = parser.parse_args()
"""
energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]

norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16, 0.0]"""


energy_edges_MESE = [10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
energy_centers_MESE = [14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]
# MESE 
#norm_segmented = [4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
#sigma_upper = [0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
#sigma_lower = [0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16, 0.0]

# Combined fit
norm_segmented = [3.36, 4.42, 2.03, 1.81, 0.089, 0.85, 0.41, 0, 0.017, 0.069]
sigma_upper = [1.10, 0.74, 0.40, 0.38, 0.350, 0.43, 0.36, 0.20, 0.073, 0.38]
sigma_lower = [0.63, 0.48, 0.39, 0.37, 0.089, 0.42, 0.25, 0.0, 0.017, 0.069]

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
#effective_area_df = pd.read_csv(effective_area_path)
# Rename the first (unnamed) column to 'energy_bins'
#effective_area_df.rename(columns={effective_area_df.columns[0]: 'energy_bins'}, inplace=True)
from Astrid.effective_area import get_effective_area_dataframe, bin_edges_to_centers

Edep = np.logspace(4, 8, 4*20+1)
energy_centers = bin_edges_to_centers(Edep)
effective_area_df = get_effective_area_dataframe(Edep, gen2=False)
effective_area_df['effective_area_m2'] = effective_area_df['nu_e'] + effective_area_df['nu_mu'] + effective_area_df['nu_tau']
print("effective area: ", effective_area_df['effective_area_m2'])
effective_area_df['energy_bins'] = energy_centers
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


def initialize_nuSIprop(Mphi_val, g_val, si_val, mntot_val, flux_model=0, si2_val=None, E_break_val=None):
    """
    Initialize nuSIprop object with given parameters.
    
    Parameters:
    -----------
    Mphi_val : float
        Mphi parameter in GeV
    g_val : float
        g parameter
    si_val : float
        Spectral index (astro_gamma)
    mntot_val : float
        mntot parameter
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(nuSIprop_path)
        evolver = nuSIprop.pyprop(
            mphi=Mphi_val * 1e6,  # Convert GeV to eV
            g=g_val,
            si=si_val,
            norm=1e-18,
            mntot=mntot_val,
            majorana=True,
            non_resonant=True,
            normal_ordering=True,
            N_bins_E=300,  # Reduced from 300 to save memory
            lEmin=12 - 0.01,
            lEmax=17.01,
            zmax=3,  # Reduced from 4 to save memory
            flav=2,
            flux_model=flux_model,
            si2=si2_val if si2_val is not None else si_val,
            E_break=1e9*E_break_val if E_break_val is not None else 1e9*1e5,
            phiphi=False,
        )
        evolver.evolve()
    finally:
        os.chdir(original_cwd)
    gc.collect()
    return evolver

"""

def get_data_bins(segmented_flux):
    #Asimov data: event counts proportional to segmented flux.
    #Units are arbitrary but consistent with model bins.

    return np.asarray(segmented_flux, dtype=float)

def get_data_uncertainty(sigma_upper, sigma_lower):

    #Event-space uncertainties corresponding to segmented flux uncertainties.
    #Units are consistent with get_data_bins() and get_model_bins().

    sigma_upp = np.asarray(sigma_upper, dtype=float)
    sigma_low = np.asarray(sigma_lower, dtype=float)
    return sigma_upp, sigma_low



def get_model_bins(model_flux_function, params, log_uniform_params=None):
    #Compute model 'event' bins using bin-integrated spectral reweighting,
    #following arXiv:2503.19960.

    # Transform log-uniform parameters back
    params_actual = params.copy()
    if log_uniform_params is not None:
        for idx in log_uniform_params:
            params_actual[idx] = np.exp(params[idx])

    N_model = np.zeros(len(energy_centers_MESE))

    for i in range(len(energy_edges_MESE) - 1):
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i + 1]

        # energy grid
        E = np.logspace(np.log10(Emin), np.log10(Emax), 200)

        # model flux
        phi_model = model_flux_function(E, params_actual)

        # reference E^{-2} integral
        ref_integral = np.trapz(E**(-2), E)

        # model integral
        model_integral = np.trapz(phi_model, E)

        # model "event count"
        N_model[i] = model_integral / ref_integral

    return N_model

"""

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


def get_model_bins(model_flux_function, params, effective_area_df, livetime, log_uniform_params=None):
    #Compute model bins from flux function. Parameters:
    #-----------
    #model_flux_function : function
    #    Function that takes (E, params) and returns flux
    #params : array
    #    Parameter array (may be in log-space for log-uniform parameters)
    #effective_area_df : DataFrame
    #    Effective area data
    #livetime : float
    #    Livetime
    #log_uniform_params : list, optional
    #    Indices of parameters that are log-uniform (need to be transformed back)
    # Transform log-uniform parameters back from log-space
    params_actual = params.copy()
    if log_uniform_params is not None:
        for idx in log_uniform_params:
            params_actual[idx] = np.exp(params[idx])
    N = np.zeros(len(energy_centers_MESE))
    for i in range(len(energy_edges_MESE)-1):   # Kanske skriva om detta så det bara beräknas en enda gång?
        Emin, Emax = energy_edges_MESE[i], energy_edges_MESE[i+1]
        E = np.logspace(np.log10(Emin), np.log10(Emax), 100)
        Aeff_i = interpolate_effective_area(effective_area_df, E)
        flux_i = model_flux_function(E, params_actual)
        integrand = Aeff_i * flux_i
        integral_i = np.trapz(integrand, E)
        N[i] = integral_i * livetime 
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
            #print('using negative sigma for bin i=', i)
            sigma =  -sigma_low[i]
            
        if energy_centers_MESE[i] < 1e5:   # 100TeV
            TS[i] = (N_data[i] - N_model[i])**2 / (sigma**2)  # Pearson's chi2 
        else:
            #print('numerator: ', i, log_likelihood_poisson(N_data[i], N_model[i]) - log_likelihood_poisson(N_data[i], N_data[i]))
            if sigma <= 0:
                #print('negative sigma')
                sigma *= (1 - 1e-4)
            
            TS[i] = 2* (log_likelihood_poisson(N_data[i], N_model[i]) - log_likelihood_poisson(N_data[i], N_data[i])) / (log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i]))
            """if i == 9:
                print('denominator: ', i, log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i]))
                print('LLH(Data|Data +sigma) =', log_likelihood_poisson(N_data[i], N_data[i] + sigma ), 'LLH(Data|Data) = ', log_likelihood_poisson(N_data[i], N_data[i]))
                print('Ndata = ', N_data[i], ', Ndata + sigma = ', sigma)
                
        if np.abs(sigma) <= 0:
            if energy_centers_MESE[i] >= 1e5:
                print('bin i = ', i)"""
            #assert log_likelihood_poisson(N_data[i], N_data[i] + sigma ) - log_likelihood_poisson(N_data[i], N_data[i]) < 0
    #print('TS final bin: ', np.round(TS[-1], 10))
    #print('final bin: ', N_data[-1], N_model[-1], sigma_upp[-1], sigma_low[-1])
    return TS

def model_flux_nusiprop(E, params, nuSIprop_obj):
    """
    nuSIprop flux model.
    
    Parameters:
    -----------
    E : array
        Energy array in GeV
    params : array
        [astro_norm, astro_gamma, Mphi, g, mntot]
    nuSIprop_obj : nuSIprop.pyprop object
        nuSIprop object (will be updated with current parameters)
    """
    #astro_norm, astro_gamma, Mphi, g, mntot = params
    #print('params', params)
    Mphi, g, mntot, astro_gamma, astro_norm = params
    
    # Convert energy from GeV to eV for nuSIprop
    E_eV = E * 1e9
    
    # Update nuSIprop parameters
    original_cwd = os.getcwd()
    try:
        os.chdir(nuSIprop_path)
        nuSIprop_obj.set_parameters(
            mphi=Mphi * 1e6,  # Convert GeV to eV
            g=g,
            si=astro_gamma,
            norm=astro_norm,
            mntot=mntot
        )
        nuSIprop_obj.evolve()
        flux_el = nuSIprop_obj.interp_flux_el(E_eV)
        flux_mu = nuSIprop_obj.interp_flux_mu(E_eV)
        flux_ta = nuSIprop_obj.interp_flux_ta(E_eV)
        flux_total = (flux_el + flux_mu + flux_ta) * 1e-18

    finally:
        os.chdir(original_cwd)      
    # Scale by astro_norm and convert units
    # nuSIprop returns flux in units that need to be scaled
    #return astro_norm * flux_total * 1e-18 / 6.0
    #print('total_flux: ', flux_total)
    return flux_total


def model_flux_nusiprop_with_shape(E, nuSIprop_obj, *, astro_norm, si, flux_model, si2=None, E_break=None, mphi=None, g=None, mntot=None):
    """
    Convenience wrapper around nuSIprop for shaped spectra (bpl/lp).
    """
    E_eV = E * 1e9
    original_cwd = os.getcwd()
    try:
        os.chdir(nuSIprop_path)
        params = dict(
            mphi=(mphi if mphi is not None else args.Mphi) * 1e6,
            g=g if g is not None else args.g,
            si=si,
            norm=astro_norm,
            mntot=mntot if mntot is not None else args.mntot,
        )
        if si2 is not None:
            params["si2"] = si2
        if E_break is not None:
            params["E_break"] = E_break*1e9
        nuSIprop_obj.set_parameters(**params)
        nuSIprop_obj.evolve()
        flux_el = nuSIprop_obj.interp_flux_el(E_eV)
        flux_mu = nuSIprop_obj.interp_flux_mu(E_eV)
        flux_ta = nuSIprop_obj.interp_flux_ta(E_eV)
        flux_total = (flux_el + flux_mu + flux_ta) * 1e-18
    finally:
        os.chdir(original_cwd)
    gc.collect()
    return flux_total


# Load effective area data (you'll need to provide this)
# For now, creating a placeholder - you'll need to load your actual effective area data
# effective_area_df = load_effective_area_data()  # You need to implement this

# Compute data bins from segmented flux
# N_data = get_data_bins(norm_segmented, effective_area_df, livetime)

# Set up parameters
# Note: astro_gamma is only used for spl, cutoff, and nusiprop models
# bpl and lp models have their own spectral index parameters
if args.model == "spl":
    parameter_names = ["astro_norm", "astro_gamma"]
    params = [args.astro_norm, args.astro_gamma]
    is_fixed = [args.fix_astro_norm, args.fix_astro_gamma]
    priors = [
        (None, None, 1.0, 5.0),  # astro_norm: uniform prior 
        (None, None, 1.0, 5.0),  # astro_gamma: uniform prior 
    ]
elif args.model == "bpl":
    parameter_names = ["astro_norm", "astro_gamma1", "astro_gamma2", "E_break"]
    params = [args.astro_norm, args.astro_gamma1, args.astro_gamma2, args.E_break]
    is_fixed = [args.fix_astro_norm, args.fix_astro_gamma1, args.fix_astro_gamma2, args.fix_E_break]
    priors = [
        (None, None, 1.0, 5.0),  # astro_norm: uniform prior 
        (None, None, 0.0, 1.0),  # astro_gamma1: uniform bounds
        (None, None, 2.4, 3.3),  # astro_gamma2: uniform bounds
        (None, None, 1e4, 1e5),  # E_break: log-uniform bounds (in GeV)
    ]
elif args.model == "lp":
    parameter_names = ["astro_norm", "alpha", "beta"]
    params = [args.astro_norm, args.alpha, args.beta]
    is_fixed = [args.fix_astro_norm, args.fix_alpha, args.fix_beta]
    priors = [
        (None, None, 1.0, 5.0),  # astro_norm: uniform prior 
        (None, None, 2.2, 3.2),  # alpha: uniform bounds
        (None, None, 0.0, 1.0),  # beta: uniform bounds 
    ]
elif args.model == "cutoff":
    parameter_names = ["astro_norm", "astro_gamma", "cutoff_energy"]
    params = [args.astro_norm, args.astro_gamma, args.cutoff_energy]
    is_fixed = [args.fix_astro_norm, args.fix_astro_gamma, args.fix_cutoff_energy]
    priors = [
        (None, None, 1.0, 5.0),  # astro_norm: uniform prior 
        (None, None, 1.0, 5.0),  # astro_gamma: uniform prior 
        (None, None, 1e5, 1e7)        
    ]
elif args.model == "nusiprop":
    parameter_names = ["Mphi", "g", "mntot", "astro_gamma", "astro_norm"]
    params = [args.Mphi, args.g, args.mntot, args.astro_gamma, args.astro_norm]
    is_fixed = [args.fix_Mphi, args.fix_g, args.fix_mntot, args.fix_astro_gamma, args.fix_astro_norm]
    priors = [
        (None, None, 0.03, 50),  # Mphi: log-uniform bounds (in GeV)
        (None, None, 1e-31, 1.0),  # g: log-uniform bounds
        (None, None, 0.06, 0.15),  # mntot: uniform bounds
        (None, None, 1.0, 5.0),  # astro_norm: uniform prior 
        (None, None, 1.0, 5.0),  # astro_gamma: uniform prior 
    ]
    

params = np.array(params)
if np.any(is_fixed):
    print("Fixing parameters")
    for b, name, val in zip(is_fixed, parameter_names, params):
        if b:
            print(name + " = ", val)

is_fitted = [not b for b in is_fixed]

# A wrapper function that handles fits with fixed parameters (similar to HESE_fit.py)
def calcLLH_fitted_func(is_fitted, params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params):
    def func(fitted_params, parameter_names, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params):
        params[:][is_fitted] = fitted_params
        ts, grads = calc_mese_ts(
            params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params
        )
        return ts, np.array(grads)[is_fitted]
    return func


def calc_mese_ts(params, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params=None, compute_grads=True):
    """
    Compute MESE test statistic (TS) for marginalization.
    
    This function uses your get_TS function to compute the test statistic.
    Uses finite differences to compute gradients.
    """
    # Compute model bins
    N_model = get_model_bins(model_flux_function, params, effective_area_df, livetime, log_uniform_params)
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
            # For log-uniform parameters, perturbation is in log-space
            if log_uniform_params is not None and i in log_uniform_params:
                params_pert[i] = params[i] + eps  # Additive in log-space
            elif params[i] != 0:
                params_pert[i] = params[i] * (1.0 + eps)
            else:
                params_pert[i] = params[i] + eps
            
            # Compute perturbed TS
            N_model_pert = get_model_bins(model_flux_function, params_pert, effective_area_df, livetime, log_uniform_params)
            TS_array_pert = get_TS(N_data, N_model_pert, sigma_upp, sigma_low)
            total_ts_pert = np.sum(TS_array_pert)
            
            # Compute gradient
            if log_uniform_params is not None and i in log_uniform_params:
                grads[i] = (total_ts_pert - total_ts) / eps
            elif params[i] != 0:
                grads[i] = (total_ts_pert - total_ts) / (params[i] * eps)
            else:
                grads[i] = (total_ts_pert - total_ts) / eps
    
    return total_ts, grads




# Select model flux function and initialize model-specific objects
if args.model == "spl":
    # Use nuSIprop with flux_model=0 (SPL)
    nuSIprop_obj = initialize_nuSIprop(args.Mphi, args.g, args.astro_gamma, args.mntot, flux_model=0)
    def model_flux_nusiprop_spl_wrapper(E, params):
        astro_norm, astro_gamma = params
        return model_flux_nusiprop_with_shape(
            E,
            nuSIprop_obj,
            astro_norm=astro_norm,
            si=astro_gamma,
            flux_model=0,
        )
    model_flux_function = model_flux_nusiprop_spl_wrapper
elif args.model == "cutoff":
    model_flux_function = model_flux_cutoff
    nuSIprop_obj = None
elif args.model == "bpl":
    # Use nuSIprop with flux_model=1, mapping gamma1/gamma2/E_break into si/si2/E_break
    nuSIprop_obj = initialize_nuSIprop(
        args.Mphi, args.g, args.astro_gamma1, args.mntot, flux_model=1, si2_val=args.astro_gamma2, E_break_val=args.E_break
    )
    def model_flux_nusiprop_bpl_wrapper(E, params):
        astro_norm, astro_gamma1, astro_gamma2, E_break = params
        return model_flux_nusiprop_with_shape(
            E,
            nuSIprop_obj,
            astro_norm=astro_norm,
            si=astro_gamma1,
            si2=astro_gamma2,
            E_break=E_break,
            flux_model=1,
        )
    model_flux_function = model_flux_nusiprop_bpl_wrapper
elif args.model == "lp":
    # Use nuSIprop with flux_model=2, mapping alpha/beta into si/si2
    nuSIprop_obj = initialize_nuSIprop(
        args.Mphi, args.g, args.alpha, args.mntot, flux_model=2, si2_val=args.beta
    )
    def model_flux_nusiprop_lp_wrapper(E, params):
        astro_norm, alpha, beta = params
        return model_flux_nusiprop_with_shape(
            E,
            nuSIprop_obj,
            astro_norm=astro_norm,
            si=alpha,
            si2=beta,
            flux_model=2,
        )
    model_flux_function = model_flux_nusiprop_lp_wrapper
elif args.model == "nusiprop":
    # Initialize nuSIprop object with initial parameters
    nuSIprop_obj = initialize_nuSIprop(
        args.Mphi, args.g, args.astro_gamma, args.mntot
    )
    # Create a wrapper function that includes nuSIprop_obj
    def model_flux_nusiprop_wrapper(E, params):
        return model_flux_nusiprop(E, params, nuSIprop_obj)
    model_flux_function = model_flux_nusiprop_wrapper
else:
    raise ValueError(f"Unknown model: {args.model}")


# Identify which parameters should be log-uniform
# cutoff_energy, E_break, Mphi, and g are log-uniform
log_uniform_params = []
if args.model == "cutoff":
    log_uniform_params.append(parameter_names.index("cutoff_energy"))
if args.model == "bpl":
    log_uniform_params.append(parameter_names.index("E_break"))
if args.model == "nusiprop":
    log_uniform_params.append(parameter_names.index("Mphi"))
    log_uniform_params.append(parameter_names.index("g"))

# Set bounds (extract from priors - only using bounds, not prior values)
# Handle log-uniform priors by using log-space bounds
bounds = []
for i, prior in enumerate(priors):
    if i in log_uniform_params:
        # For log-uniform, use log-space bounds
        bounds.append((np.log(prior[2]), np.log(prior[3])))
    else:
        bounds.append((prior[2], prior[3]))
bounds = np.array(bounds)

# Transform log-uniform parameters to log-space for optimization
if len(log_uniform_params) > 0:
    params_transformed = params.copy()
    for idx in log_uniform_params:
        if params[idx] > 0:
            params_transformed[idx] = np.log(params[idx])
        else:
            # Use lower bound if parameter is not positive
            params_transformed[idx] = np.log(priors[idx][2])
    params = params_transformed



# Compute N_data from segmented flux
N_data = get_data_bins(norm_segmented, effective_area_df, livetime)
print("N_data: ", N_data)
sigma_upp, sigma_low = get_data_uncertainty(sigma_upper, sigma_lower, effective_area_df, livetime)
print("sigma_upp: ", sigma_upp)
print("sigma_low: ", sigma_low)
# Run the fit
calcLLH = calcLLH_fitted_func(is_fitted, np.copy(params),
                              effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params)
gc.collect()
start = time.time()
print("Running fit")

print("params[is_fitted]: ", params[is_fitted])

fitted_params, ts, info = fmin_l_bfgs_b(
    calcLLH,
    x0=params[is_fitted],
    args=(parameter_names, effective_area_df, livetime, N_data, model_flux_function, sigma_upp, sigma_low, log_uniform_params),
    bounds=bounds[is_fitted],
    m=10,
    pgtol=1e-10,
    factr=1e7,
)

end = time.time()

print("Fit took " + str(end - start) + " seconds")
BF_params = params[:]
BF_params[is_fitted] = fitted_params

print("Best Fit TS: ", ts)
print("Best Fit Parameters:")
# Transform log-uniform parameters back from log-space for display
BF_params_display = BF_params.copy()
for idx in log_uniform_params:
    BF_params_display[idx] = np.exp(BF_params[idx])

for param, BF_param, BF_param_display in zip(parameter_names, BF_params, BF_params_display):
    if param in [parameter_names[i] for i in log_uniform_params]:
        print("\t{}: \t{} (exp({}))".format(param, BF_param_display, BF_param))
    else:
        print("\t{}: \t{}".format(param, BF_param))

print(info)
print(ts)
# Output parameters in original space (not log-space)
print(BF_params_display.tolist())