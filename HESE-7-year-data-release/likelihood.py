import numpy as np
import scipy as sp
import scipy.special

import autodiff
import weighter


def gammaPriorPoissonLikelihood(k, alpha, beta):
    """Poisson distribution marginalized over the rate parameter, priored with
       a gamma distribution that has shape parameter alpha and inverse rate
       parameter beta.
    Parameters
    ----------
    k : int
        The number of observed events
    alpha : tuple
        Zeroeth element contains Gamma distribution shape parameter
        First element contains respective gradients
    beta : tuple
        Zeroeth element contains Gamma distribution inverse rate parameter
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    val = autodiff.mul_grad(alpha, autodiff.log(beta))
    val = autodiff.plus_grad(val, autodiff.lgamma(autodiff.plus(alpha, k)))
    val = autodiff.minus(val, sp.special.loggamma(k + 1.0))
    val = autodiff.minus_grad(
        val, autodiff.mul_grad(autodiff.plus(alpha, k), autodiff.log1p(beta))
    )
    val = autodiff.minus_grad(val, autodiff.lgamma(alpha))

    return val


def poissonLikelihood(k, weight_sum):
    """Computes Log of the Poisson Likelihood.
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """

    logw = autodiff.log(weight_sum)

    klogw = autodiff.mul_r(k, logw)

    klogw_minus_w = autodiff.minus_grad(klogw, weight_sum)

    llh = autodiff.minus(klogw_minus_w, sp.special.loggamma(k + 1))

    return llh


def LEff(k, weight_sum, weight_sq_sum):
    """Computes Log of the L_Eff Likelihood.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weight_sum : 2D tuple
        Zeroeth element contains the sum of the weighted MC event counts
        First element contains respective gradients
    weight_sq_sum : 2D tuple
        Zeroeth element containsthe sum of the square of the weighted MC event counts
        First element contains respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """

    # Return -inf for an ill formed likelihood or 0 without observation
    if weight_sum[0] <= 0 or weight_sq_sum[0] < 0:
        # weight_sum[1] should be a 1D array with gradients (one per parameter)
        # After autodiff.sum, it should be shape (n_params,)
        if k == 0:
            return (0.0, np.zeros_like(weight_sum[1]))
        else:
            return (-np.inf, np.zeros_like(weight_sum[1]))

    # Return the poisson likelihood in the appropriate limiting case
    if weight_sq_sum[0] == 0:
        return poissonLikelihood(k, weight_sum)

    alpha = autodiff.plus(
        autodiff.div_grad(autodiff.pow(weight_sum, 2), weight_sq_sum), 1.0
    )
    beta = autodiff.div_grad(weight_sum, weight_sq_sum)
    L = gammaPriorPoissonLikelihood(k, alpha, beta)
    return L


def computeLEff(k, weights):
    """Computes Log of the L_Eff Likelihood from a list of weights.
       This is the poisson likelihood, using a poisson distribution with
       rescaled rate parameter to describe the Monte Carlo expectation, and
       assuming a uniform prior on the rate parameter of the Monte Carlo.
       This is the main result of the paper arXiv:1901.04645
    Parameters
    ----------
    k : int
        the number of observed events
    weights : 2D tuple
        Zeroeth element contains list of the weighted MC events
        First element contains list of respective gradients
    Returns
    -------
    2D tuple
        Zeroeth element contains the log likelihood
        First element contains respective gradients
    """
    weight_sum = autodiff.sum(weights)
    weight_sq_sum = autodiff.sum(autodiff.pow(weights, 2))

    return LEff(k, weight_sum, weight_sq_sum)


def calcEffLLH(data, weights, bin_slices):
    """
    Computes and returns the effective log likelihood
    Parameters
    -----------
    data: array-like
        list of observed events in each analysis bin.
    weights: array-like
        list of sorted weights.
    bin_slices: array-like
        list of bin slices, where each slice picks out the elements in weights
        corresponding to an analysis bin.

    Returns
    --------
    tuple:
        Zeroth element is the effective log likelihood
        First element is the gradient of the effective log likelihood
    """
    llhs = []
    
    for i, bin_slice in enumerate(bin_slices):
        if bin_slice.stop - bin_slice.start == 0:
            continue
        llhs.append(
            computeLEff(data[i], (weights[0][bin_slice], weights[1][bin_slice]))
        )
    
    # Ensure all gradient arrays have consistent shape
    # Extract scalar LLH values and gradient arrays
    llh_values = [llh[0] for llh in llhs]
    llh_grads = [np.atleast_1d(llh[1]).flatten() for llh in llhs]
    
    # Check that all gradients have the same shape
    """if len(llh_grads) > 0:
        expected_shape = llh_grads[0].shape
        for i, grad in enumerate(llh_grads):
            if grad.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent gradient shapes: bin {i} has shape {grad.shape}, "
                    f"expected {expected_shape}. This may indicate a problem with "
                    f"parameter gradients in different bins."
                )"""
    
    llhs = (np.array(llh_values), np.array(llh_grads))
    llh = autodiff.sum(llhs)

    return llh


def calcLLH(
    params, parameter_names, priors, bin_slices, data, weighter_maker, livetime,
    exclude_prior_indices=None, original_data=None, apply_reco_corrections=False
):
    """
    Computes and returns the total negative log likelihood

    Parameters
    ----------
    exclude_prior_indices : list of int, optional
        Indices of parameters to exclude from prior calculation
        (e.g., for profile likelihood tests on fixed parameters)
    original_data : array, optional
        Original uncorrected data (needed if apply_reco_corrections=True)
    apply_reco_corrections : bool, optional
        If True, apply reconstruction corrections and re-bin data

    Returns
    --------
    tuple:
        Zeroth element is the total negative log likelihood
        First element is the gradient of the -llh
    """
    if exclude_prior_indices is None:
        exclude_prior_indices = []
    
    # Apply reconstruction corrections if needed
    # Note: bin_slices are for MC, data is binned_data (counts per bin)
    # We need to re-bin the data to match the MC bin structure
    if apply_reco_corrections:
        if original_data is None:
            raise ValueError("original_data must be provided when apply_reco_corrections=True")
        import reco_corrections
        import binning
        
        # Get reconstruction correction parameters
        if "reco_energy_scale" in parameter_names:
            energy_scale_idx = parameter_names.index("reco_energy_scale")
            energy_scale = params[energy_scale_idx]
        else:
            energy_scale = 1.0
        
        if "reco_zenith_shift" in parameter_names:
            zenith_shift_idx = parameter_names.index("reco_zenith_shift")
            zenith_shift = params[zenith_shift_idx]
        else:
            zenith_shift = 0.0
        
        # Apply corrections and re-bin DATA (not MC - MC is never modified)
        # Note: We need to use the same binning as MC, so we use bin_data which
        # uses the same bin structure. The bin_slices parameter is for MC weights,
        # so we keep it unchanged. We just update the data counts.
        # IMPORTANT: original_data is never modified - apply_reco_corrections creates a copy
        corrected_data = reco_corrections.apply_reco_corrections(
            original_data, energy_scale=energy_scale, zenith_shift=zenith_shift
        )
        sorted_data, data_bin_slices = binning.bin_data(corrected_data)
        
        # Count events in each bin (this is what 'data' should be)
        # Note: data_bin_slices should match mc_bin_slices in structure
        # (same number of bins, same bin edges)
        binned_data = np.array([len(sorted_data[dbin]) for dbin in data_bin_slices])
        
        # Update data (binned counts)
        # bin_slices stays the same (it's for MC weights)
        data = binned_data
    elif data is None:
        raise ValueError("data must be provided when apply_reco_corrections=False")

    weights = weighter_maker.get_weights(livetime, parameter_names, params)

    PriorLLH = [0.0, np.zeros(shape=len(params)).astype(float)]

    # The loop calculates and adds the prior llh to the effective llh
    for i, (param, prior) in enumerate(zip(params, priors)):
        
        # Skip prior if this parameter index should be excluded
        if i in exclude_prior_indices:
            continue

        # Prior format: (mu, sigma, low, high) or (mu, sigma, low, high, "log_uniform")
        # or (mu, sigma, low, high, None) for uniform priors with 5 elements
        # Check if "log_uniform" is in the prior tuple
        is_log_uniform = "log_uniform" in prior
        
        # Always extract first 4 elements (mu, sigma, low, high)
        # The 5th element (if present) is either "log_uniform" or None
        mu, sigma, low, high = prior[0], prior[1], prior[2], prior[3]

        if param > high or param < low:
            PriorLLH = (-np.inf, np.zeros(shape=len(params)).astype(float))
            break
        
        # Skip uniform priors (no prior contribution)
        if mu == None and not is_log_uniform:
            continue

        grad = np.zeros(shape=len(params)).astype(float)
        grad[i] = 1.0

        if is_log_uniform:
            # Log-uniform prior: log(p(x)) = -log(x) + constant
            # This gives uniform sampling in log space
            LLH = autodiff.mul(autodiff.log((param, grad)), -1.0)
            PriorLLH = autodiff.plus_grad(PriorLLH, LLH)
        elif mu != None:
            # Gaussian prior
            LLH = autodiff.normal_log_pdf((param, grad), mu, sigma)
            PriorLLH = autodiff.plus_grad(PriorLLH, LLH)

    EffLLH = calcEffLLH(data, weights, bin_slices)

    # Combine the effective llh and prior llh, and take the negative
    return autodiff.mul(autodiff.plus_grad(PriorLLH, EffLLH), -1.0)
