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


def HESE_effective_areas(json_files=MC_FILENAMES, energy_bins=np.logspace(2, 7, 5 * 20 + 1), interaction_channels=False):
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
    def get_eff(masks, factor=1.0):
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
           
           
    # The factor of 0.5 is needed so that we compute the average
    # neutrino/antineutrino effective area. This is in contrast to the
    # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
    # sum of the neutrino and antineutrino effective areas.
    eff, eff_err = [], []
    
    if interaction_channels == False:
        print('test')
        for flavor_index, flavor in enumerate(["e", "mu", "tau"]):

            particle_key = "2nu" + flavor
            particle_mask = particle_masks[particle_key]
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            
            eff_f, eff_err_f = get_eff(masks, factor=1)
            eff.append(eff_f)
            eff_err.append(eff_err_f)
        #print(eff.shape)
        eff = [eff[0], eff[1], eff[2]]  # Dont want to distinguish between particle/antiparticle
        #for eff_ in eff:
        #    eff_ = [2*x for x in eff_]    #  Double the effective area to account for particle/antiparticle
    
    
    elif interaction_channels == True:
        for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
            for interaction_index, interaction in enumerate(["CC", "NC", "GR"]):
            
                particle_key = "2nu" + flavor
                particle_mask = particle_masks[particle_key]
                particle_mask = np.logical_and(
                    particle_mask, interaction_masks[interaction_index]
                )
                masks = np.logical_and(particle_mask[None, :], nu_energy_masks)

                eff_f, eff_err_f = get_eff(masks, factor=1)
                eff.append(eff_f)
                eff_err.append(eff_err_f)
            
            
            
    """    
    else:
            
        # Make an effective area vs. energy plot split by flavor and interaction type
        # Charged Current per flavor
        for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
            for interaction_index, interaction in enumerate(["CC"]):

                label = (
                    f"ν_{flavor if flavor == 'e' else '\\' + flavor}"
                    + f" + ν̄_{flavor if flavor == 'e' else '\\' + flavor}"
                    + f" {interaction}"
                )
                particle_key = "2nu" + flavor
                particle_mask = particle_masks[particle_key]
                particle_mask = np.logical_and(
                    particle_mask, interaction_masks[interaction_index]
                )
                masks = np.logical_and(particle_mask[None, :], nu_energy_masks)

                eff_f, eff_err_f = get_eff(masks, label, factor=1)
                eff.append(eff_f)
                eff_err.append(eff_err_f)

        # Neutral Current All Flavor
        label = "NC All Flavor"
        masks = np.logical_and(interaction_masks[1][None, :], nu_energy_masks)
        #plot_line(ax, masks, color, line_style, label, factor=0.5)

        # Glashow Resonance
        flavor = "e"
        label = (
            f"ν_{flavor}"
            + f" + ν̄_{flavor}"
            + " GR"
        )
        particle_key = "2nu" + flavor
        particle_mask = particle_masks[particle_key]
        particle_mask = np.logical_and(particle_mask, interaction_masks[2])
        masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
        #plot_line(ax, masks, color, line_style, label, factor=0.5)"""

    
    print("Final eff shape:", np.array(eff).shape)
    #print("Final eff_err shape:", np.array(eff_err).shape)
    # Oklart hur göra med eff_err
    return eff, eff_err

    


def get_effective_area_range(eff, Edep, gen2=False):
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
    energy_bins = np.logspace(2, 7, 5 * 20 + 1)  # Original energy bins

    energy_bins = energy_bins[1:]
    emin, emax = Edep[0], Edep[-1]

    # Filter energy bins within the range of Edep
    mask = (energy_bins > emin) & (energy_bins <= min(emax, 1e7))
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
        print('No extrapolation needed')
        energy_extrapolated = []
        energy_combined = energy_filtered
        eff_new = [area[mask] * (10 if gen2 else 1) for area in eff]

    #print(len(eff_new[0]), len(energy_combined))
    return eff_new, energy_combined



def get_effective_area_dataframe(Edep, gen2=False, interaction_channels=False):
    # Compute limited/extrapolated effective area and energy bins 
    eff, eff_err = HESE_effective_areas(interaction_channels=interaction_channels)
    print("eff shape:", np.asarray(eff).shape)
    
    eff_new_range, energy_bins_new = get_effective_area_range(eff, Edep, gen2=gen2)
    print("eff_new_range shape:", np.asarray(eff_new_range).shape)
    print("energy_bins_new shape:", energy_bins_new.shape)
    
    # Convert bin edges to centers
    energy_bins_new = np.concatenate(([Edep[0]], energy_bins_new))
    energy_centers = bin_edges_to_centers(energy_bins_new)
    
    eff_new_range = np.asarray(eff_new_range)
    
    if interaction_channels == False:
        eff_df = pd.DataFrame(eff_new_range.T, index=energy_centers, columns=['nu_e', 'nu_mu', 'nu_tau'])
    else:
        eff_df = pd.DataFrame(eff_new_range.T, index=energy_centers, columns=['nu_e_CC', 'nu_e_NC', 'nu_e_GR', 'nu_mu_CC', 'nu_mu_NC', 'nu_mu_GR', 'nu_tau_CC', 'nu_tau_NC', 'nu_tau_GR'])
    print("eff_df shape:", eff_df.shape)
    return eff_df


def total_events(flx, eff, livetime, norm, delta_E, save_to_csv=False):
    # Interpolate `flx` to the same energy bins as `eff`
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)
    

    total_events_df = flx_interpolated * eff * livetime * norm 
    if total_events_df.index.min() < 0:
        print('total_events_df.index.min() < 0')
        negative_mask = total_events_df < 0
        total_events_df[negative_mask] = 0
        
    #total_events_df = flx * eff_interpolated * livetime
    total_events_df['total_events'] = delta_E * (total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau'])
    #total_events_df.index = eff.index 

    if save_to_csv==True:
        total_events_df.to_csv('total_events.csv')

    return total_events_df


def bin_centers_to_edges(bin_centers):
    log_centers = np.log10(bin_centers)
    dlog = np.diff(log_centers)
    log_edges = np.zeros(len(bin_centers) + 1)
    log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
    log_edges[0] = log_centers[0] - dlog[0]/2
    log_edges[-1] = log_centers[-1] + dlog[-1]/2
    return 10**log_edges


def bin_edges_to_centers(bin_edges):
    log_edges = np.log10(bin_edges)
    dlog = np.diff(log_edges)
    log_centers = np.zeros(len(bin_edges) - 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    return 10**log_centers


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
    smeared_events = np.zeros_like(events)  
    
    for i, E_true in enumerate(energies):
        # Calculate sigma in linear space (resolution is fractional)
        sigma = resolution * E_true
        
        # Create Gaussian in linear space
        gaussian = np.exp(-0.5 * ((energies - E_true) / sigma) ** 2)
        
        # Normalize the Gaussian
        gaussian_sum = np.sum(gaussian)
        if gaussian_sum > 0:  # Avoid division by zero
            gaussian /= gaussian_sum
        
        # Redistribute events
        smeared_events += events[i] * gaussian
    
    # Verify event conservation
    total_events_before = np.sum(events)
    total_events_after = np.sum(smeared_events)
    if not np.isclose(total_events_before, total_events_after, rtol=1e-10):
        print(f"Warning: Event conservation violated! Before: {total_events_before}, After: {total_events_after}")
    
    return smeared_events


def test_energy_smearing():
    """
    Test function to verify the energy smearing implementation.
    """
    # Test 1: Single peak
    energies = np.logspace(4, 7, 100)
    events = np.zeros_like(energies)
    events[50] = 1000  # All events in one bin
    
    smeared = apply_energy_smearing(energies, events, resolution=0.1)
    
    # Verify event conservation
    assert np.isclose(np.sum(events), np.sum(smeared), rtol=1e-10), "Event conservation failed"
    
    # Test 2: Multiple peaks
    events = np.zeros_like(energies)
    events[30] = 500  # First peak
    events[70] = 500  # Second peak
    
    smeared = apply_energy_smearing(energies, events, resolution=0.1)
    
    # Verify event conservation
    assert np.isclose(np.sum(events), np.sum(smeared), rtol=1e-10), "Event conservation failed"
    
    # Test 3: Resolution dependence
    smeared_high_res = apply_energy_smearing(energies, events, resolution=0.2)
    smeared_low_res = apply_energy_smearing(energies, events, resolution=0.05)
    
    # Calculate spread in linear space
    high_res_spread = np.sqrt(np.average((energies - np.average(energies, weights=smeared_high_res))**2, weights=smeared_high_res))
    low_res_spread = np.sqrt(np.average((energies - np.average(energies, weights=smeared_low_res))**2, weights=smeared_low_res))
    
    # Higher resolution should result in more spread
    assert high_res_spread > low_res_spread, "Resolution effect not working as expected"
    
    print("All energy smearing tests passed!")


def visualize_energy_smearing():
    """
    Create visualizations to verify the energy smearing behavior.
    """
    import matplotlib.pyplot as plt
    
    # Create test data
    energies = np.logspace(4, 7, 100)
    events = np.zeros_like(energies)
    events[50] = 1000  # Single peak
    
    # Apply smearing with different resolutions
    smeared_10 = apply_energy_smearing(energies, events, resolution=0.1)
    smeared_20 = apply_energy_smearing(energies, events, resolution=0.2)
    smeared_05 = apply_energy_smearing(energies, events, resolution=0.05)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.step(energies, events, where='mid', label='Original', color='black')
    plt.step(energies, smeared_05, where='mid', label='5% resolution', color='blue', alpha=0.7)
    plt.step(energies, smeared_10, where='mid', label='10% resolution', color='red', alpha=0.7)
    plt.step(energies, smeared_20, where='mid', label='20% resolution', color='green', alpha=0.7)
    
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('Energy Smearing Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Test with multiple peaks
    events = np.zeros_like(energies)
    events[30] = 500  # First peak
    events[70] = 500  # Second peak
    
    smeared_10 = apply_energy_smearing(energies, events, resolution=0.1)
    
    plt.figure(figsize=(10, 6))
    plt.step(energies, events, where='mid', label='Original', color='black')
    plt.step(energies, smeared_10, where='mid', label='Smeared (10% resolution)', color='red', alpha=0.7)
    
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('Energy Smearing Test - Multiple Peaks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

