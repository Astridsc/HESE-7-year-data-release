"""
Script to compute the test statistic TS(E_cutoff) for the cutoff model vs SPL.

This implements a PROFILE LIKELIHOOD RATIO TEST:

TS(E_cutoff) = -2 * log((max_{eta} LL_cutoff(E_cutoff, eta) * prior(eta)) / 
                         (max_{eta} LL_SPL(eta) * prior(eta)))

Where:
- E_cutoff is the PARAMETER OF INTEREST (what we're testing)
- η are NUISANCE PARAMETERS (all other parameters: astro_gamma, astro_norm, etc.)

For each fixed E_cutoff:
1. Fix E_cutoff at that value
2. Maximize over all nuisance parameters η (fit the model)
3. Compare to SPL model (which has no E_cutoff)

Since calcLLH returns -[LL + log_prior], we have:
TS(E_cutoff) = 2 * [best_neg_LLH_SPL - best_neg_LLH_cutoff]

Why test multiple E_cutoff values?
- To find the best-fit E_cutoff: test many values and pick the one with maximum TS
- To create a profile likelihood curve: see how TS varies with E_cutoff
- To test a specific hypothesis: test only that one E_cutoff value
"""

import sys
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, minimize_scalar
import argparse

import weighter
import binning
import data_loader
import likelihood


def fit_model(model_type, fixed_params_dict, parameter_names, priors, 
              mc_bin_slices, binned_data, weight_maker, livetime, initial_params):
    """
    Fit a model with some parameters fixed.
    
    Parameters:
    -----------
    model_type : str
        "spl" or "cutoff"
    fixed_params_dict : dict
        Dictionary of parameter_name: value for fixed parameters
    parameter_names : list
        List of all parameter names
    priors : list
        List of prior tuples
    initial_params : array
        Initial parameter values
    
    Returns:
    --------
    best_neg_LLH : float
        Best negative log likelihood (including priors)
    best_params : array
        Best fit parameters
    """
    
    # Create weight maker with appropriate model
    weight_maker_model = weighter.Weighter(
        weight_maker.mc, 
        nuSIprop=False, 
        model=model_type
    )
    
    # Determine which parameters are fixed
    is_fixed = [name in fixed_params_dict for name in parameter_names]
    is_fitted = [not b for b in is_fixed]
    
    # Set fixed parameter values
    params = initial_params.copy()
    for name, value in fixed_params_dict.items():
        idx = parameter_names.index(name)
        params[idx] = value
    
    # Determine which parameter indices to exclude from prior calculation
    # For profile likelihood: exclude prior on fixed parameters (e.g., E_cutoff)
    # For SPL model: exclude prior on cutoff_energy (not a parameter of SPL)
    exclude_prior_indices = []
    if model_type == "spl":
        # SPL model doesn't have E_cutoff, so exclude its prior
        if "cutoff_energy" in parameter_names:
            exclude_prior_indices.append(parameter_names.index("cutoff_energy"))
    else:
        # For cutoff model, exclude prior on fixed parameters
        for name in fixed_params_dict.keys():
            if name in parameter_names:
                exclude_prior_indices.append(parameter_names.index(name))
    
    # Wrapper function for fitting
    def calcLLH_fitted_func(is_fitted, params, exclude_prior_indices):
        def func(fitted_params, parameter_names, priors, mc_bin_slices, 
                 binned_data, weights, livetime):
            params[:][is_fitted] = fitted_params
            llh, grads = likelihood.calcLLH(
                params, parameter_names, priors, mc_bin_slices,
                binned_data, weights, livetime,
                exclude_prior_indices=exclude_prior_indices
            )
            return llh, np.array(grads[0])[is_fitted]
        return func
    
    calcLLH = calcLLH_fitted_func(is_fitted, params.copy(), exclude_prior_indices)
    
    # Get bounds
    bounds = np.array([(prior[2], prior[3]) for prior in priors])
    
    # Handle epsilon_dom bimodality
    bounds_list = []
    if "epsilon_dom" not in fixed_params_dict:
        index = parameter_names.index("epsilon_dom")
        bounds_low = bounds.copy()
        bounds_high = bounds.copy()
        bounds_low[index] = [0.8, 0.99]
        bounds_high[index] = [0.99, 1.25]
        bounds_list = [bounds_low, bounds_high]
    else:
        bounds_list = [bounds]
    
    # Fit for each set of bounds
    fitted_params_list = []
    llh_list = []
    
    for bounds in bounds_list:
        fitted_params, llh, info = fmin_l_bfgs_b(
            calcLLH,
            x0=params[is_fitted],
            args=(parameter_names, priors, mc_bin_slices, binned_data,
                  weight_maker_model, livetime),
            bounds=bounds[is_fitted],
            m=10,
            pgtol=1e-18,
            factr=1e4,
        )
        fitted_params_list.append(fitted_params)
        llh_list.append(llh)
    
    # Get best fit
    min_index = np.argmin(llh_list)
    best_neg_LLH = llh_list[min_index]
    best_fitted_params = fitted_params_list[min_index]
    
    # Reconstruct full parameter vector
    best_params = params.copy()
    best_params[is_fitted] = best_fitted_params
    
    return best_neg_LLH, best_params


def compute_TS_at_cutoff(E_cutoff, best_neg_LLH_spl, initial_params, parameter_names, priors,
                        mc_bin_slices, binned_data, weight_maker_base, livetime):
    """
    Compute TS at a specific E_cutoff value.
    This is used for optimization to find the minimum TS.
    
    Returns:
    --------
    TS : float
        Test statistic at this E_cutoff
    """
    # Fix cutoff_energy
    fixed_params = {"cutoff_energy": E_cutoff}
    
    # Use initial parameters as starting guess (set E_cutoff appropriately)
    initial_params_cutoff = initial_params.copy()
    initial_params_cutoff[parameter_names.index("cutoff_energy")] = E_cutoff
    
    best_neg_LLH_cutoff, _ = fit_model(
        "cutoff", fixed_params, parameter_names, priors,
        mc_bin_slices, binned_data, weight_maker_base, livetime, initial_params_cutoff
    )
    
    # TS = 2 * (best_neg_LLH_cutoff - best_neg_LLH_spl)
    TS = 2.0 * (best_neg_LLH_cutoff - best_neg_LLH_spl)
    
    return TS


def main():
    parser = argparse.ArgumentParser(
        description="Compute test statistic TS(E_cutoff) for cutoff model vs SPL. "
                    "This is a PROFILE LIKELIHOOD test: for each E_cutoff (parameter of interest), "
                    "we maximize over all nuisance parameters η (astro_gamma, astro_norm, etc.)."
    )
    parser.add_argument("--cutoff_energy", type=float, default=None,
                       help="Single cutoff energy value in GeV to test")
    parser.add_argument("--cutoff_energy_values", type=str, default=None,
                       help="Comma-separated list of cutoff energies to test")
    parser.add_argument("--find_best_cutoff", action="store_true",
                       help="Find best-fit E_cutoff by optimizing over it (in addition to nuisance params)")
    parser.add_argument("--scan_cutoff_range", type=str, default=None,
                       help="Scan E_cutoff range: 'min,max,npoints' (e.g., '1e6,1e7,20') to find maximum TS")
    parser.add_argument("--optimize_TS", action="store_true",
                       help="Optimize to find E_cutoff that minimizes TS (best cutoff fit)")
    parser.add_argument("--optimize_start", type=float, default=4e6,
                       help="Starting E_cutoff value for optimization (default: 4e6)")
    
    args = parser.parse_args()
    
    if args.cutoff_energy is None and args.cutoff_energy_values is None and not args.find_best_cutoff and args.scan_cutoff_range is None and not args.optimize_TS:
        parser.error("Must specify either --cutoff_energy, --cutoff_energy_values, --find_best_cutoff, --scan_cutoff_range, or --optimize_TS")
    
    livetime = 227708167.68
    
    # Parameter setup (same as HESE_fit.py)
    parameter_names = [
        "cr_delta_gamma", "nunubar_ratio", "anisotropy_scale",
        "astro_gamma", "astro_norm", "conv_norm",
        "epsilon_dom", "epsilon_head_on", "muon_norm",
        "kpi_ratio", "prompt_norm", "beta", "cutoff_energy",
    ]
    
    # Initial parameters: use SPL best-fit values for faster convergence
    # These are the best-fit parameters from HESE_fit.py for the SPL model
    initial_params = np.array([
        -0.05309198828568753,  # cr_delta_gamma
        0.998164256210702,     # nunubar_ratio
        1.0007247919049886,    # anisotropy_scale
        2.8737764773857943,    # astro_gamma
        6.365300091182592,     # astro_norm
        1.006210702376819,     # conv_norm
        0.9519225902130987,    # epsilon_dom
        -0.05499094382686424,  # epsilon_head_on
        1.1868488857218278,    # muon_norm
        1.0001423496123587,    # kpi_ratio
        0.0,                   # prompt_norm
        0.0,                   # beta (not used in SPL)
        0.0                    # cutoff_energy (will be set for each fit)
    ])
    
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
        (None, None, 1e5, 1e7, "log_uniform"),  # cutoff_energy
    ]
    
    # Load data
    mc_filenames = [
        "./resources/data/HESE_mc_observable.json",
        "./resources/data/HESE_mc_flux.json",
        "./resources/data/HESE_mc_truth.json",
    ]
    mc = data_loader.load_mc(mc_filenames)
    data = data_loader.load_data("./resources/data/HESE_data.json")
    
    sorted_mc, mc_bin_slices = binning.bin_data(mc)
    sorted_data, data_bin_slices = binning.bin_data(data)
    binned_data = np.array([len(sorted_data[data_bin]) for data_bin in data_bin_slices])
    
    # Create weight maker (model doesn't matter here, we'll create new ones)
    weight_maker_base = weighter.Weighter(sorted_mc, nuSIprop=False, model="spl")
    
    # SPL best-fit negative log-likelihood (hardcoded since it's constant)
    best_neg_LLH_spl = 122.959199
    print(f"Using SPL best-fit -LLH: {best_neg_LLH_spl:.6f}")
    print()
    
    # Determine which cutoff energies to test
    if args.scan_cutoff_range:
        # Parse range: min,max,npoints
        parts = args.scan_cutoff_range.split(',')
        if len(parts) != 3:
            raise ValueError("--scan_cutoff_range must be 'min,max,npoints' (e.g., '1e6,1e7,20')")
        E_min = float(parts[0].strip())
        E_max = float(parts[1].strip())
        npoints = int(parts[2].strip())
        # Create log-spaced grid
        cutoff_energies = np.logspace(np.log10(E_min), np.log10(E_max), npoints)
        print(f"Scanning {npoints} E_cutoff values from {E_min:.2e} to {E_max:.2e} GeV (log-spaced)")
    elif args.find_best_cutoff:
        # Find best-fit E_cutoff by optimizing over it
        print("Finding best-fit E_cutoff by optimizing over all parameters...")
        best_neg_LLH_cutoff_full, best_params_cutoff_full = fit_model(
            "cutoff", {}, parameter_names, priors,
            mc_bin_slices, binned_data, weight_maker_base, livetime, initial_params
        )
        best_E_cutoff = best_params_cutoff_full[parameter_names.index("cutoff_energy")]
        print(f"Best-fit E_cutoff: {best_E_cutoff:.2e} GeV")
        print(f"Best -LLH (cutoff, full fit): {best_neg_LLH_cutoff_full:.6f}")
        print()
        
        print("Computing test statistics...")
        print("=" * 60)
        
        # Compute TS for the best-fit (E_cutoff optimized as free parameter)
        # TS = -2 * log(L_cutoff / L_spl) = 2 * (best_neg_LLH_cutoff - best_neg_LLH_spl)
        TS_full = 2.0 * (best_neg_LLH_cutoff_full - best_neg_LLH_spl)
        print(f"TS (full optimization, E_cutoff as free parameter): {TS_full:.6f}")
        print(f"  (TS > 0 favors SPL, TS < 0 favors cutoff)")
        
        # For p-value, we need |TS| since chi2 distribution is for TS^2
        # But wait - if TS < 0, that means cutoff is better, so we want to test
        # the significance of the improvement. The p-value should be based on |TS|
        from scipy.stats import chi2
        TS_abs = abs(TS_full)
        if TS_abs > 0:
            # p-value is probability of observing |TS| or larger under null (SPL)
            p_value_full = 1.0 - chi2.cdf(TS_abs, df=1)
            print(f"p-value (chi2, df=1, using |TS|={TS_abs:.6f}): {p_value_full:.4f}")
        else:
            print("Note: TS = 0 means models are equivalent")
        print()
        
        # Also compute profile TS at the best-fit value for comparison
        cutoff_energies = [best_E_cutoff]
        print("Computing profile TS at best-fit E_cutoff for comparison...")
    elif args.optimize_TS:
        # Simple line search: walk upward from starting point until TS starts increasing
        print(f"Searching for minimum TS starting from E_cutoff = {args.optimize_start:.2e} GeV")
        print("Walking upward until TS minimum is found...")
        print("=" * 60)
        
        # Get bounds from prior
        cutoff_idx = parameter_names.index("cutoff_energy")
        cutoff_prior = priors[cutoff_idx]
        if "log_uniform" in cutoff_prior:
            prior_list = [p for p in cutoff_prior if p != "log_uniform"]
            _, _, E_min, E_max = prior_list
        else:
            _, _, E_min, E_max = cutoff_prior
        
        # Step size: use logarithmic steps (since E_cutoff spans orders of magnitude)
        # Start with larger steps, then refine
        step_factor = 1.1  # 10% steps in log space
        max_steps = 20
        tolerance = 1e4  # Stop when step is less than 10 TeV
        
        current_E = args.optimize_start
        prev_TS = None
        best_E = current_E
        best_TS = None
        results_opt = []
        
        for step in range(max_steps):
            # Compute TS at current E_cutoff
            TS = compute_TS_at_cutoff(
                current_E, best_neg_LLH_spl, initial_params, parameter_names, priors,
                mc_bin_slices, binned_data, weight_maker_base, livetime
            )
            print(f"  Step {step+1}: E_cutoff = {current_E:.2e} GeV, TS = {TS:.6f}")
            results_opt.append({"E_cutoff": current_E, "TS": TS})
            
            # Track best (minimum TS)
            if best_TS is None or TS < best_TS:
                best_TS = TS
                best_E = current_E
            
            # Check if we've passed the minimum (TS is increasing)
            if prev_TS is not None and TS > prev_TS:
                print(f"  TS increased from {prev_TS:.6f} to {TS:.6f}, minimum found!")
                break
            
            # Check if we've hit the upper bound
            if current_E >= E_max * 0.99:
                print(f"  Reached upper bound ({E_max:.2e} GeV)")
                break
            
            # Move to next point
            prev_TS = TS
            current_E = current_E * step_factor
            
            # If step becomes too small, stop
            step_size = current_E - results_opt[-1]["E_cutoff"]
            if step_size < tolerance:
                print(f"  Step size ({step_size:.2e} GeV) below tolerance")
                break
        
        print()
        print(f"Search complete!")
        print(f"Best E_cutoff: {best_E:.2e} GeV")
        print(f"Minimum TS: {best_TS:.6f}")
        print()
        
        # Compute p-value using |TS|
        from scipy.stats import chi2
        TS_abs = abs(best_TS)
        if TS_abs > 0:
            p_value = 1.0 - chi2.cdf(TS_abs, df=1)
            print(f"p-value (chi2, df=1, using |TS|={TS_abs:.6f}): {p_value:.4f}")
        else:
            print("Note: TS = 0 means models are equivalent")
        print()
        
        # Use the best E_cutoff for detailed output
        cutoff_energies = [best_E]
        print("Computing detailed fit at best-fit E_cutoff...")
    elif args.cutoff_energy_values:
        cutoff_energies = [float(x.strip()) for x in args.cutoff_energy_values.split(",")]
    else:
        cutoff_energies = [args.cutoff_energy]
    
    
    # Now fit cutoff model for each E_cutoff value
    # Use SPL best-fit as initial guess for faster convergence
    results = []
    for E_cutoff in cutoff_energies:
        print(f"Fitting cutoff model with E_cutoff = {E_cutoff:.2e} GeV...")
        
        # Fix cutoff_energy
        fixed_params = {"cutoff_energy": E_cutoff}
        
        # Use initial parameters as starting guess (set E_cutoff appropriately)
        initial_params_cutoff = initial_params.copy()
        initial_params_cutoff[parameter_names.index("cutoff_energy")] = E_cutoff
        
        best_neg_LLH_cutoff, best_params_cutoff = fit_model(
            "cutoff", fixed_params, parameter_names, priors,
            mc_bin_slices, binned_data, weight_maker_base, livetime, initial_params_cutoff
        )
        
        # Compute test statistic
        # Paper's definition: TS = -2 * log(L_cutoff / L_spl)
        # Since calcLLH returns -[LL + log_prior], we have:
        # log L = -best_neg_LLH
        # So: TS = -2 * (log L_cutoff - log L_spl)
        #     = -2 * (-best_neg_LLH_cutoff - (-best_neg_LLH_spl))
        #     = -2 * (-best_neg_LLH_cutoff + best_neg_LLH_spl)
        #     = 2 * (best_neg_LLH_cutoff - best_neg_LLH_spl)
        # TS > 0 means SPL is better (L_spl > L_cutoff)
        # TS < 0 means cutoff is better (L_cutoff > L_spl)
        print(f"Computing test statistics for E_cutoff = {E_cutoff:.2e} GeV...")
        print("=" * 60)
        TS = 2.0 * (best_neg_LLH_cutoff - best_neg_LLH_spl)
        
        print(f"  Best -LLH (cutoff): {best_neg_LLH_cutoff:.6f}")
        print(f"  TS(E_cutoff = {E_cutoff:.2e}): {TS:.6f}")
        print()
        
        results.append({
            "E_cutoff": E_cutoff,
            "TS": TS,
            "neg_LLH_cutoff": best_neg_LLH_cutoff,
            "neg_LLH_spl": best_neg_LLH_spl,
            "params_cutoff": best_params_cutoff,
        })
    
    # Print summary
    print("=" * 60)
    print("Summary:")
    print(f"{'E_cutoff (GeV)':<20} {'TS':<15} {'-LLH_cutoff':<15} {'-LLH_SPL':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['E_cutoff']:<20.2e} {r['TS']:<15.6f} {r['neg_LLH_cutoff']:<15.6f} {r['neg_LLH_spl']:<15.6f}")
    
    # Find minimum TS (most negative = best fit for cutoff model)
    # and maximum |TS| for p-value calculation
    if len(results) > 1:
        min_TS_idx = np.argmin([r['TS'] for r in results])
        min_result = results[min_TS_idx]
        max_abs_TS_idx = np.argmax([abs(r['TS']) for r in results])
        max_abs_result = results[max_abs_TS_idx]
        
        print()
        print(f"Minimum TS (best cutoff fit): {min_result['TS']:.6f} at E_cutoff = {min_result['E_cutoff']:.2e} GeV")
        print(f"Maximum |TS| (for p-value): {max_abs_result['TS']:.6f} at E_cutoff = {max_abs_result['E_cutoff']:.2e} GeV")
        
        # Compute p-value using |TS| (chi-square distribution is for TS^2, but we use |TS|)
        # p-value is probability of observing |TS| or larger under null hypothesis (SPL)
        from scipy.stats import chi2
        TS_abs = abs(max_abs_result['TS'])
        if TS_abs > 0:
            p_value = 1.0 - chi2.cdf(TS_abs, df=1)
            print(f"p-value (chi2, df=1, using |TS|={TS_abs:.6f}): {p_value:.4f}")
        else:
            print("Note: TS = 0 means models are equivalent")
    
    # Also compute p-value for single result
    elif len(results) == 1:
        from scipy.stats import chi2
        TS_abs = abs(results[0]['TS'])
        if TS_abs > 0:
            p_value = 1.0 - chi2.cdf(TS_abs, df=1)
            print()
            print(f"p-value (chi2, df=1, using |TS|={TS_abs:.6f}): {p_value:.4f}")
    
    return results


if __name__ == "__main__":
    results = main()

