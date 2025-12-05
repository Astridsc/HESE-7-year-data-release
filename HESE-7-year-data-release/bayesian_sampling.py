"""
Bayesian sampling script using ultranest for model comparison.

This script can be used for:
- Single parameter of interest (e.g., E_cutoff)
- Multiple parameters of interest (e.g., nuSIprop with Mphi, g, mntot, si, norm)

Usage:
    python bayesian_sampling.py --model cutoff --params_of_interest cutoff_energy
    python bayesian_sampling.py --model nusiprop --params_of_interest Mphi,g,mntot,si,norm
"""

import numpy as np
import sys
import os

# Add ultranest import (install with: pip install ultranest)
try:
    import ultranest
    from ultranest import ReactiveNestedSampler
except ImportError:
    print("Error: ultranest not installed. Install with: pip install ultranest")
    sys.exit(1)

import weighter
import binning
import data_loader
import likelihood


def compute_log_likelihood(params_dict, model_type, parameter_names, priors,
                          mc_bin_slices, binned_data, weight_maker_base, livetime):
    """
    Compute log-likelihood for a given set of parameters.
    
    This is the function that ultranest will call repeatedly.
    It evaluates the full likelihood (not profile likelihood).
    
    Parameters:
    -----------
    params_dict : dict
        Dictionary of parameter_name: value
    model_type : str
        "spl", "cutoff", "nusiprop", etc.
    parameter_names : list
        List of all parameter names
    priors : list
        List of prior tuples
    mc_bin_slices, binned_data, weight_maker_base, livetime : 
        Data and setup objects
        
    Returns:
    --------
    log_likelihood : float
        Log-likelihood (not negative log-likelihood)
    """
    # Convert dict to array in the correct order
    params_array = np.array([params_dict[name] for name in parameter_names])
    
    # Create weight maker with appropriate model
    weight_maker_model = weighter.Weighter(
        weight_maker_base.mc,
        nuSIprop=weight_maker_base.nuSIprop if hasattr(weight_maker_base, 'nuSIprop') else False,
        model=model_type
    )
    
    # Compute negative log-likelihood (including priors)
    neg_llh, _ = likelihood.calcLLH(
        params_array, parameter_names, priors, mc_bin_slices,
        binned_data, weight_maker_model, livetime
    )
    
    # Return log-likelihood (ultranest expects positive log-likelihood)
    return -neg_llh


def prior_transform(cube, params_to_sample, parameter_names, priors, default_values=None):
    """
    Transform unit cube [0,1]^n to physical parameter space.
    
    Parameters:
    -----------
    cube : array
        Unit cube values [0,1]^n where n = len(params_to_sample) or len(parameter_names)
    params_to_sample : list or None
        List of parameter names to sample. If None, samples all parameters.
    parameter_names : list
        List of all parameter names
    priors : list
        List of prior tuples
    default_values : dict, optional
        Default values for parameters not being sampled (if params_to_sample is not None)
        
    Returns:
    --------
    params_dict : dict
        Dictionary of all parameters
    """
    params_dict = {}
    
    # Determine which parameters to sample
    if params_to_sample is None:
        # Sample all parameters
        params_to_sample = parameter_names
    
    # Transform each parameter being sampled
    for i, param_name in enumerate(params_to_sample):
        if param_name not in parameter_names:
            raise ValueError(f"Unknown parameter: {param_name}")
        
        idx = parameter_names.index(param_name)
        prior = priors[idx]
        
        # Check if log-uniform prior
        is_log_uniform = "log_uniform" in prior
        if is_log_uniform:
            prior_list = [p for p in prior if p != "log_uniform"]
            mu, sigma, low, high = prior_list
        else:
            mu, sigma, low, high = prior
        
        # Transform from unit cube to parameter space
        u = cube[i]
        
        if is_log_uniform:
            # Log-uniform: uniform in log space
            param_value = low * (high / low) ** u
        elif mu is not None:
            # Gaussian prior: use inverse CDF
            from scipy.stats import norm
            param_value = norm.ppf(u, loc=mu, scale=sigma)
            # Clip to bounds
            param_value = np.clip(param_value, low, high)
        else:
            # Uniform prior
            param_value = low + (high - low) * u
        
        params_dict[param_name] = param_value
    
    # Set default values for parameters not being sampled (if any)
    if params_to_sample is not None and default_values is not None:
        for param_name in parameter_names:
            if param_name not in params_dict:
                if param_name in default_values:
                    params_dict[param_name] = default_values[param_name]
                else:
                    params_dict[param_name] = 0.0
    
    return params_dict


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bayesian sampling using ultranest for model comparison"
    )
    parser.add_argument("--model", type=str, required=True,
                       choices=["spl", "cutoff", "nusiprop"],
                       help="Model to sample")
    parser.add_argument("--params_of_interest", type=str, default=None,
                       help="Comma-separated list of parameters of interest (for marginalization). If None, samples all parameters.")
    parser.add_argument("--sample_all", action="store_true",
                       help="Sample over all parameters (parameters of interest + nuisance parameters)")
    parser.add_argument("--n_live_points", type=int, default=400,
                       help="Number of live points for nested sampling (default: 400)")
    parser.add_argument("--min_num_live_points", type=int, default=40,
                       help="Minimum number of live points (default: 40)")
    
    args = parser.parse_args()
    
    # Determine which parameters to sample
    if args.sample_all:
        # Sample all parameters (proper Bayesian approach)
        print("Sampling over ALL parameters (parameters of interest + nuisance parameters)")
        params_to_sample = None  # Will be set to all parameters
        params_of_interest = args.params_of_interest.split(",") if args.params_of_interest else None
    elif args.params_of_interest:
        # Only sample parameters of interest (nuisance parameters fixed - profile-like)
        params_to_sample = [p.strip() for p in args.params_of_interest.split(",")]
        params_of_interest = params_to_sample
        print(f"WARNING: Only sampling parameters of interest. Nuisance parameters will be fixed at default values.")
        print(f"This is more like profile likelihood than true Bayesian sampling.")
    else:
        parser.error("Must specify either --params_of_interest or --sample_all")
    
    print(f"Bayesian sampling for {args.model} model")
    if params_of_interest:
        print(f"Parameters of interest (for marginalization): {params_of_interest}")
    print("=" * 60)
    
    # Setup (same as compute_TS_cutoff.py)
    livetime = 227708167.68
    
    parameter_names = [
        "cr_delta_gamma", "nunubar_ratio", "anisotropy_scale",
        "astro_gamma", "astro_norm", "conv_norm",
        "epsilon_dom", "epsilon_head_on", "muon_norm",
        "kpi_ratio", "prompt_norm", "beta", "cutoff_energy",
    ]
    
    # Add nuSIprop parameters if needed
    if args.model == "nusiprop":
        parameter_names.extend(["Mphi", "g", "mntot", "si"])
    
    priors = [
        (-0.05, 0.05, -np.inf, np.inf),  # cr_delta_gamma
        (1.0, 0.1, 0.0, 2.0),            # nunubar_ratio
        (1.0, 0.2, 0.0, 2.0),            # anisotropy_scale
        (None, None, -np.inf, np.inf),   # astro_gamma
        (None, None, 0.0, np.inf),        # astro_norm
        (1.0, 0.4, 0.0, np.inf),         # conv_norm
        (0.99, 0.1, 0.8, 1.25),          # epsilon_dom
        (0.0, 0.5, -3.82, 2.18),         # epsilon_head_on
        (1.0, 0.5, 0.0, np.inf),         # muon_norm
        (1.0, 0.1, 0.0, np.inf),         # kpi_ratio
        (None, None, 0.0, np.inf),       # prompt_norm
        (None, None, -np.inf, np.inf),   # beta
        (None, None, 1e5, 1e7, "log_uniform"),  # cutoff_energy
    ]
    
    # Add nuSIprop priors if needed
    if args.model == "nusiprop":
        priors.extend([
            (None, None, 0.03, 100, "log_uniform"),    # Mphi
            (None, None, 1e-4, 1.0, "log_uniform"),   # g
            (None, None, 0.06, 0.15, None),           # mntot (uniform)
            (None, None, 2.0, 3.0, None),             # si (uniform)
        ])
    
    # Determine which parameters to actually sample
    if params_to_sample is None:
        # Sample all parameters (proper Bayesian approach)
        params_to_sample = parameter_names
        default_values = None
    else:
        # Only sample specified parameters, fix others at defaults
        default_values = {
            "cr_delta_gamma": -0.05309198828568753,
            "nunubar_ratio": 0.998164256210702,
            "anisotropy_scale": 1.0007247919049886,
            "astro_gamma": 2.8737764773857943,
            "astro_norm": 6.365300091182592,
            "conv_norm": 1.006210702376819,
            "epsilon_dom": 0.9519225902130987,
            "epsilon_head_on": -0.05499094382686424,
            "muon_norm": 1.1868488857218278,
            "kpi_ratio": 1.0001423496123587,
            "prompt_norm": 0.0,
            "beta": 0.0,
            "cutoff_energy": 0.0,
            "Mphi": 0.0,
            "g": 0.0,
            "mntot": 0.0,
            "si": 0.0,
        }
    
    n_dim = len(params_to_sample)
    print(f"Number of dimensions to sample: {n_dim}")
    print(f"Parameters being sampled: {params_to_sample}")
    print("=" * 60)
    
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
    
    # Create weight maker base
    weight_maker_base = weighter.Weighter(sorted_mc, nuSIprop=False, model=args.model)
    
    # Initialize nuSIprop if needed
    if args.model == "nusiprop":
        # You'll need to initialize nuSIprop here
        # This is a placeholder - adjust based on your setup
        import nuSIprop
        weight_maker_base.nuSIprop = nuSIprop.pyprop(
            mphi=5*1e6, g=0.1, si=2.5, norm=1e-18, mntot=0.1,
            majorana=True, non_resonant=True, normal_ordering=True,
            N_bins_E=300, lEmin=13, lEmax=16.01, zmax=5, flav=2, phiphi=False
        )
    
    # Define likelihood function for ultranest
    def log_likelihood(cube):
        """Log-likelihood function for ultranest"""
        params_dict = prior_transform(cube, params_to_sample, parameter_names, priors, default_values)
        return compute_log_likelihood(
            params_dict, args.model, parameter_names, priors,
            mc_bin_slices, binned_data, weight_maker_base, livetime
        )
    
    # Define prior transform
    def transform(cube):
        """Prior transform for ultranest"""
        params_dict = prior_transform(cube, params_to_sample, parameter_names, priors, default_values)
        # Return as array in the order of params_to_sample
        return np.array([params_dict[p] for p in params_to_sample])
    
    # Run nested sampling
    print("Starting nested sampling...")
    sampler = ReactiveNestedSampler(
        params_to_sample,
        log_likelihood,
        transform=transform,
        log_dir="./ultranest_output/",
        resume="overwrite"
    )
    
    result = sampler.run(
        min_num_live_points=args.min_num_live_points,
        dlogz=0.5,  # Stopping criterion
        max_num_improvement_loops=2,
    )
    
    sampler.print_results()
    sampler.plot()
    
    print("\nSampling complete!")
    print(f"Evidence (log Z): {result['logz']:.2f} ± {result['logzerr']:.2f}")
    
    return result


if __name__ == "__main__":
    result = main()

