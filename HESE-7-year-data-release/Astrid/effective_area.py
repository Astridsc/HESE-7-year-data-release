#import sys
#import os
#import os.path
#base_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, base_path + "/resources/external/")

#import matplotlib
#import matplotlib.style
#matplotlib.use("TkAgg")
#matplotlib.style.use("./resources/mpl/paper.mplstyle")
#import matplotlib.pyplot as plt
#from matplotlib.font_manager import FontProperties
#from matplotlib.collections import LineCollection
#import functools

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import pandas as pd
import binning
import json


from Astrid.config import MC_FILENAMES
from Astrid.data_processing import load_true_events, get_particle_masks


def center(x):
    x = np.asarray(x)
    return (x[1:] + x[:-1]) / 2.0


def HESE_effective_areas(json_files=MC_FILENAMES, energy_bins=np.logspace(2, 7, 5 * 20 + 1)):
    # Load the MC
    json_data = dict()
    for filename in json_files:
        json_data.update(json.load(open(filename, "r")))

    # Get the MC generation information
    weight_over_flux_over_livetime = np.array(json_data["weightOverFluxOverLivetime"])

    # We are going to average our effective area over the whole sky
    energy_bin_widths = np.diff(energy_bins)
    total_angular_width = 4.0 * np.pi
    bin_widths = energy_bin_widths * total_angular_width

    # Get neutrino interaction information from the file
    primaryEnergy = np.array(json_data["primaryEnergy"])
    interactionType = np.array(json_data["interactionType"])
    primaryType = np.array(json_data["primaryType"])

     # Get some masks that correspond to our chosen energy bins
    nu_energy_mapping = np.digitize(primaryEnergy, bins=energy_bins) - 1
    nu_energy_masks = [nu_energy_mapping == i for i in range(len(energy_bins) - 1)]

    # Get some masks that sort by interaction type
    interaction_types = [1, 2, 3]
    interaction_masks = [interactionType == i for i in interaction_types]
    CC_mask, NC_mask, GR_mask = interaction_masks

    # Get some masks that sort by primary particle type
    # Remember these are the relevant entries in the dictionary:
    """
        'nue',   'nuebar',   '2nue',
        'numu',  'numubar',  '2numu',
        'nutau', 'nutaubar', '2nutau',
        'mu', 'nu', 'all',
    """
    particle_masks = get_particle_masks(primaryType)

    # How to compute and plot the effective area (in a histogram style with errors)
    def get_eff(masks, label, factor=1.0):
        # Each entry in masks corresponds to an energy bin
        # The mask should define the events that contribute to the effective area calcualtion in that bin

        # Effective area is the sum of weightOverFluxOverLivetime, divided by bin width
        effective_area_cm2 = np.array(
            [
                np.sum(weight_over_flux_over_livetime[mask]) / bin_width
                for mask, bin_width in zip(masks, bin_widths)
            ]
        ) * factor
        # An additional factor may be needed if we are computing an average
        # effective area for multiple particle types

        # Compute the error on this quantity
        effective_area_cm2_error = np.array(
            [
                np.sqrt(np.sum(weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                for mask, bin_width in zip(masks, bin_widths)
            ]
        ) * factor

        # Convert to meters^2
        meter = 100
        effective_area_m2 = effective_area_cm2 / (meter ** 2)
        effective_area_m2_error = effective_area_cm2_error / (meter ** 2)

        return effective_area_m2, effective_area_m2_error
           
    # Let's make an effective area vs. energy plot split by neutrino flavor
    #fig, ax = plt.subplots(figsize=(7, 5))
    eff, eff_err = [], []
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        label = f"ν_{flavor} + ν̄_{flavor}"
        particle_key = "2nu" + flavor
        particle_mask = particle_masks[particle_key]
        masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
        # The factor of 0.5 is needed so that we compute the average
        # neutrino/antineutrino effective area. This is in contrast to the
        # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
        # sum of the neutrino and antineutrino effective areas.
        eff_f, eff_err_f = get_eff(masks, label, factor=1)
        eff.append(eff_f)
        eff_err.append(eff_err_f)

    eff = [eff[0], eff[1], eff[2]]  # Dont want to distinguish between particle/antiparticle
    for eff_ in eff:
        eff_ = [2*x for x in eff_]    #  Double the effective area to account for particle/antiparticle
    
    # Oklart hur göra med eff_err
    return eff, eff_err


def get_effective_area_range(eff, Edep, gen2=True):
    """
    Get the effective area for a specified energy range (Edep) and extrapolate if necessary.

    Parameters:
    eff : list of numpy arrays
        Effective area values for different flavors.
    Edep : numpy array
        Desired energy range (e.g., np.logspace(emin, emax, n)).
    gen2 : bool, optional
        Scale the effective area by a factor of 10 for Gen2. Default is True.

    Returns:
    eff_new : list of numpy arrays
        Effective area values for the specified energy range.
    energy_bins : numpy array
        Energy bins corresponding to the effective area.
    """
    # Original energy bins as seen in the effective area plots by HESE
    energy_bins = np.logspace(2, 7, 5 * 20)  # Original energy bins

    emin, emax = Edep[0], Edep[-1]

    # Filter energy bins within the range of Edep
    mask = (energy_bins > emin) & (energy_bins <= min(emax, 1e7))
    #print('mask: ', mask)
    energy_filtered = energy_bins[mask]

    # Extrapolate for Edep[-1] > 1e7
    if emax > 1e7 + 10:
        delta_e = np.log10(emax) - np.log10(1e7)
        num_bins = int(delta_e) * 20  # Number of bins to extrapolate
        # Combine filtered and extrapolated energy bins
        energy_extrapolated = np.logspace(7, np.log10(emax), num=num_bins) 
        energy_combined = np.concatenate((energy_filtered, energy_extrapolated))

        # Combine the filtered area and projected area for each flavor
        projected_eff = (2* 10 ** (0.29 * np.log10(energy_extrapolated) - 0.38))
        eff_new = [
            np.concatenate((area[mask], projected_eff)) * (10 if gen2 else 1)
            for area in eff
        ]
    else:
        energy_extrapolated = []
        energy_combined = energy_filtered
        eff_new = [area[mask] * (10 if gen2 else 1) for area in eff]

    #print(len(eff_new[0]), len(energy_combined))
    return eff_new, energy_combined


def get_effective_area_dataframe(Edep, gen2=True):
    # Compute limited/extrapolated effective area and energy bins 
    eff, eff_err = HESE_effective_areas()
    eff_new_range, energy_bins_new = get_effective_area_range(eff, Edep, gen2=gen2)
    eff_new_range = np.asarray(eff_new_range)
    eff_df = pd.DataFrame(eff_new_range.T, index=energy_bins_new, columns=['nu_e', 'nu_mu', 'nu_tau'])
    return eff_df


def total_events(flx, eff, livetime, norm, delta_E, save_to_csv=False):
    # Interpolate `flx` to the same energy bins as `eff`
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)

    total_events_df = flx_interpolated * eff * livetime * norm 
    #total_events_df = flx * eff_interpolated * livetime
    total_events_df['total_events'] = delta_E * (total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau'])
    #total_events_df.index = eff.index 

    if save_to_csv==True:
        total_events_df.to_csv('total_events.csv')

    return total_events_df


def rebinning_old(total_events, Edep):
    # Same procedure as used by HESE
    emin, emax = Edep[0], Edep[-1]
    nbins = len(Edep)
    width = (np.log10(emax) - np.log10(emin)) / nbins
    e_edges, _, _ = binning.get_bins(emin, emax, ewidth=width, eedge=emin)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:]))) # Oklart varför dom börjar med index 1 istället för 0

    # Group data into logarithmic bins
    total_events_binned = total_events.groupby(pd.cut(total_events.index, bin_centers, include_lowest=True)).sum()
    #total_events_binned = pd.cut(total_events['total'], e_edges, include_lowest=True)
    #print('total_events_binned: ', total_events_binned)

    # Compute the midpoint (center) of each logarithmic interval
    # Geometric mean for log step midpoints
    total_events_binned['interval_center'] = [
        (interval.left * interval.right) ** 0.5 for interval in total_events_binned.index
    ]
    #total_events_binned.index = total_events_binned['interval_center']
    #total_events_binned['interval_center'] = bin_centers

    return total_events_binned



def bin_centers_to_edges(bin_centers):
    log_centers = np.log10(bin_centers)
    dlog = np.diff(log_centers)
    log_edges = np.zeros(len(bin_centers) + 1)
    log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
    log_edges[0] = log_centers[0] - dlog[0]/2
    log_edges[-1] = log_centers[-1] + dlog[-1]/2
    return 10**log_edges


def rebinning(total_events, Edep):
    """
    Rebin total_events to new bins specified by Edep (bin centers).
    Groups by the nearest bin center in Edep.
    """
    # Assign each event/bin to the nearest Edep bin center
    bin_indices = np.digitize(total_events.index, Edep) - 1  # -1 to get 0-based index

    # Create a DataFrame with the bin index and original data
    df = total_events.copy()
    df['bin_index'] = bin_indices

    # Group by bin_index and sum
    grouped = df.groupby('bin_index').sum()
    print(len(grouped))
    print('grouped: ', grouped)
    #grouped = grouped.replace(0, 1e-10)

    # Set the index to the corresponding Edep bin center
    grouped.index = Edep[grouped.index]
    #grouped = grouped.reindex(range(len(Edep) - 1), fill_value=0)

    return grouped


def apply_energy_smearing(energies, events, resolution):
    """
    Apply Gaussian smearing in energy space to redistribute events.
    
    Parameters:
        energies (array-like): Energy bin centers.
        events (array-like): Number of events in each energy bin.
        resolution (float): Detector resolution (fractional, e.g., 0.1 for 10%).
        
    Returns:
        smeared_events (np.ndarray): The smeared event distribution.
    """
    smeared_events = np.zeros_like(events)  # Initialize array for smeared events
    
    #bin_width = energies[1] - energies[0]  # Assuming uniform binning
    #bin_widths = np.diff(energies)  # Calculate bin widths for non-uniform bins
    
    for i, E_true in enumerate(energies):
        # Gaussian width depends on resolution and energy
        logE = np.log10(energies)
        logE_true = np.log10(E_true)
        sigma_log = resolution  # Now resolution is fractional in log10(E)
        gaussian = np.exp(-0.5 * ((logE - logE_true) / sigma_log) ** 2)
        #sigma = resolution * E_true  
        #gaussian = np.exp(-0.5 * ((energies - E_true) / sigma) ** 2)

        gaussian_sum = np.sum(gaussian)
        gaussian /= gaussian_sum  # Normalize Gaussian for proper redistribution

        # Redistribute current bin's events according to the Gaussian
        smeared_events += events[i] * gaussian
        if i == 0 or i == len(energies)-1:
            print(f"Edge bin {i}: gaussian sum = {gaussian_sum}")
    
    return smeared_events
# Check code is normalized to 1 for the gaussian 
# check for different sigmas/resolution 


def bin_weights(mc, weights, e_edges):
    """
    Bin the MC events and their weights according to energy bins.

    Parameters:
        mc: dict or DataFrame with key 'recoDepositedEnergy'
        weights: array-like, weights for each event
        e_edges: array-like, bin edges for deposited energy
        bin_centers: array-like, bin centers for deposited energy

    Returns:
        pd.DataFrame: DataFrame with columns ['bin_center', 'sum_weights', 'n_events']
    """
    # Get deposited energies and weights
    energies = mc['recoDepositedEnergy']
    weights = np.asarray(weights)

    # Bin the weights: sum weights in each energy bin
    sum_weights, _ = np.histogram(energies, bins=e_edges, weights=weights)
    n_events, _ = np.histogram(energies, bins=e_edges)
    #print(len(e_edges), len(bin_centers))
    print(len(sum_weights), len(n_events), len(e_edges))

    # Build DataFrame
    df = pd.DataFrame({
        'edges': e_edges[1:],
        'sum_weights': sum_weights,
        'total_events': n_events
    })
    """
    print('length of e_edges', len(e_edges))
    print('length of sum_weights', len(sum_weights))
    print('length of n_events', len(n_events))
    df = pd.DataFrame({
        'edges': e_edges[:-1],
        'sum_weights': sum_weights,
        'total_events': n_events
    })"""

    return df
