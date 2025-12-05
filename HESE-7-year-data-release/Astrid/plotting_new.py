import sys
import os
import os.path
sys.path.append('/home/astridaurora/nuSIprop')

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import matplotlib
import matplotlib.style
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import numpy as np
import json
import pandas as pd

import data_loader
import weighter
import binning

from Astrid.config import MC_FILENAMES, PARAMETER_NAMES, PARAMS
from Astrid.config import LIVETIME1, LIVETIME2, LIVETIME3


from Astrid.data_processing import get_weights, get_data
from Astrid.effective_area import total_events, bin_centers_to_edges
from Astrid.effective_area import apply_energy_smearing, get_effective_area_dataframe
from Astrid.save_mc_histograms import get_mc_histogram



def Enu_to_Edep(flux):
    """
    Convert the neutrino energy flux to deposited energy flux.
    """
    # Fractions of events that are cascades, tracks, and double cascades according to the HESE paper
    cascade = 0.727
    tracks = 0.234
    double_cascade = 0.039
    # Energy resolutions for each type of event
    cascade_res = 0.11
    track_res = 0.30
    double_cascade_res = 0.18
    
    
def load_hese_data():
    """
    Load the HESE data.
    """
    data = json.load(open(os.path.join(base_path, '../resources/data/HESE_data.json'), 'r'))
    data_df = pd.DataFrame(data)
    data_df.to_csv('HESE_data.csv')
    return data_df


def fig8_new():
    """
    Plot Figure 8 from the HESE paper.
    """

    energy_bins = np.logspace(5, 8, 3 * 20 + 1)
    energy_bins_new = np.logspace(5, 8, 31)

    # Get the effective area
    eff_df = get_effective_area_dataframe(energy_bins, gen2=True)

    # Get nuSIprop flux data
    flx_df = pd.read_csv(os.path.join(base_path, 'flux/flux_Fig8_600bins.csv'), index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    bin_centers = flx_df.index.values
    bin_edges = bin_centers_to_edges(bin_centers)
    delta_E = np.diff(bin_edges)

    total_events_df = total_events(flx=eff_df, eff=flx_df, norm=1.5*1e-13, livetime=LIVETIME2, delta_E=delta_E)
    
    # Get MC events and bin them equally to the nuSIprop flux bins
    mc, weights = get_weights(energy_bins, livetime=LIVETIME2, gen2=True)
    mc_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=bin_edges, weights=weights[0])
    mc_df = pd.DataFrame({
        'bin_centers': bin_centers,
        'events': mc_events,
    })
    mc_df.set_index('bin_centers', inplace=True)
    
    # Apply energy smearing
    total_events_df['with_resolution'] = apply_energy_smearing(
        energies=np.asarray(total_events_df.index), 
        events=np.asarray(total_events_df['total_events']),
        resolution=0.1
        )

    mc_df['with_resolution'] = apply_energy_smearing( 
        energies=np.asarray(mc_df.index), 
        events=np.asarray(mc_df['events']), 
        resolution=0.1)
    
    print(f"mc_df: {mc_df}")
    print(f"Total events: {total_events_df}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 9),
                            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.hist(
        total_events_df.index.values, 
        weights=total_events_df['with_resolution'], 
        bins=energy_bins_new,
        label='nuSIprop',
        histtype='step',
        color='r'
    )
    ax1.hist(
        total_events_df.index.values, 
        weights=total_events_df['total_events'], 
        bins=energy_bins_new,
        label='without resolution',
        histtype='step',
        color='lightgray',
        linestyle='--'
    )
    cm = plt.get_cmap("magma")
    ax1.hist(
        mc_df.index.values, 
        weights=mc_df['with_resolution'], 
        bins=energy_bins_new,
        label='MC HESE events',
        histtype='bar', 
        color=cm(75)
    )

    """flx_hist, _ = np.histogram(total_events_df.index.values, bins=energy_bins_new, weights=total_events_df['with_resolution'])
    mc_hist, _ = np.histogram(mc_df.index.values, bins=energy_bins_new, weights=mc_df['with_resolution'])
    bin_centers, _ = np.histogram(total_events_df.index.values, bins=energy_bins_new)"""
    # Remove zero values in `flx_hist` to avoid division by zero during normalization
    """nonzero = flx_hist > 0
    print(len(flx_hist), len(nonzero))
    mc_hist = mc_hist[nonzero]
    flx_hist = flx_hist[nonzero]"""
    #bin_centers_nonzero = mc_df.index.values[nonzero]

    # Compute normalized difference
    #normalized_diff = (flx_hist - mc_hist) / np.sqrt(flx_hist)
    #normalized_diff = (total_events_df['with_resolution'] - mc_df['with_resolution']) / np.sqrt(total_events_df['with_resolution'])

    ax2.hist(
        total_events_df.index.values,  # Bin centers corresponding to non-zero elements
        bins=energy_bins_new,
        weights=normalized_diff,
        histtype="step",
        label=r"Normalized Difference",
    color="b",
    )

    ax1.loglog()
    plt.xlim(2*energy_bins_new[0], energy_bins_new[-1])
    ax1.set_ylim(4*1.0e-1, 1.0e2)
    ax1.set_ylabel("Number of events")
    ax1.yaxis.set_ticks_position('both')      # Show ticks on both left and right
    ax1.tick_params(axis='y', which='both', right=True, labelright=False)  # Enable right ticks, hide right labels
    ax1.grid(True, which='both', axis='both', alpha=0.3, color='lightgrey')
    ax1.set_title("IceCube Gen2, 10 years livetime")
    ax1.legend()
    ax2.set_ylim(-5, 5)
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$\Delta N/\sqrt{\Delta N_{\nu SI}}$')
    ax2.set_xlabel(r"$E_{dep} [GeV]$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def fig6_new():
    """
    Plot Figure 6 from the HESE paper.
    """
    livetime = 227708167.68
    energy_bins = np.logspace(4, 7, 3 * 20 + 1)  # For initial effective area calculation, consistent with HESE original bins
    energy_bins_new = np.logspace(4, 7, 28)      # For final plotting
    
    # Get the effective area
    # Index = bin edges
    eff_df = get_effective_area_dataframe(energy_bins)
    print(f"eff_df: {eff_df}")

    # Get nuSIprop flux data
    # Index = bin centers
    flx_df = pd.read_csv(os.path.join(base_path, 'flux/flux_Fig6_600bins.csv'), index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]
    bin_centers = flx_df.index.values
    bin_edges = bin_centers_to_edges(bin_centers)
    delta_E = np.diff(bin_edges)
    print("Flux data bins:", len(bin_centers))

    # Calculate total events
    
    total_events_df = total_events(flx=eff_df, eff=flx_df, norm=0.8*1e-13, livetime=livetime, delta_E=delta_E)
    # Get MC events using the updated save_mc_histograms function
    mc_df = get_mc_histogram(Edep=energy_bins_new, livetime=livetime, gen2=False, save=False)
    print(f"mc_df: {mc_df}")
    mc_df.set_index('energy', inplace=True)
    
    # Apply energy smearing
    total_events_df['with_resolution'] = apply_energy_smearing(
        energies=np.asarray(total_events_df.index), 
        events=np.asarray(total_events_df['total_events']),
        resolution=0.1)

    """mc_df['with_resolution'] = apply_energy_smearing( 
        energies=np.asarray(mc_df.index), 
        events=np.asarray(mc_df['events']), 
        resolution=0.15)"""
    print(energy_bins_new)
    print(f"mc_df: {mc_df}")
    print(f"Total events: {total_events_df}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.hist(
        total_events_df.index.values, 
        weights=total_events_df['total_events'], 
        bins=energy_bins_new,
        label='nuSIprop (no smearing)',
        histtype='step',
        color='lightgray'
    )
    plt.hist(
        mc_df.index.values, 
        weights=mc_df['events'], 
        bins=energy_bins_new,
        label='MC events (no smearing)',
        histtype='step',
        color='lightgray',
        linestyle='--'
    )
    
    # Plot smeared data
    plt.hist(
        total_events_df.index.values,
        weights=total_events_df['with_resolution'], 
        bins=energy_bins_new, 
        label='nuSIprop', 
        histtype='step', 
        color='blue'
    )
    plt.hist(  
        mc_df.index.values,
        weights=mc_df['with_resolution'], 
        bins=energy_bins_new, 
        label='MC Events', 
        histtype='step', 
        color='orange'
    )
    
    plt.xscale('log')
    plt.xlim(6*energy_bins_new[0], energy_bins_new[-1])
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('Fig 6')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    

    
def test_morph():
    mc, weights = get_weights(np.logspace(4, 7, 3 * 20 + 1), livetime=227708167.68, gen2=False)
    json_files=MC_FILENAMES
    json_data = dict()
    for filename in json_files:
        json_data.update(json.load(open(filename, "r")))
        
    primaryEnergy = np.array(json_data["primaryEnergy"])
    #interactionType = np.array(json_data["interactionType"])
    primaryType = np.array(json_data["primaryType"])
    print(f"Primary energy: {primaryEnergy}")
    print(primaryType)
    nu_e_events = primaryType == 0
    nu_mu_events = primaryType == 1
    nu_tau_events = primaryType == 2
    morphology = np.array(json_data["recoMorphology"])
    cascades = np.array(json_data["recoDepositedEnergy"])[morphology == 0]
    tracks = np.array(json_data["recoDepositedEnergy"])[morphology == 1]
    double_cascades = np.array(json_data["recoDepositedEnergy"])[morphology == 2]
    
    
    """
    bins = np.logspace(4, 7, 30)
    plt.hist(primaryEnergy[morphology == 0], bins=bins, histtype='step', label='Cascades primary')
    plt.hist(primaryEnergy[morphology == 1], bins=bins, histtype='step', label='Tracks primary')
    plt.hist(primaryEnergy[morphology == 2], bins=bins, histtype='step', label='Double Cascades primary')
    plt.hist(cascades, bins=bins, histtype='step', label='Cascades deposited energy')
    plt.hist(tracks, bins=bins, histtype='step', label='Tracks deposited energy')
    plt.hist(double_cascades, bins=bins, histtype='step', label='Double Cascades deposited energy')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('Morphology of MC events')
    plt.legend()
    plt.show()"""
    

    
def fig6_morphologies():
    """
    Plot Figure 6 from the HESE paper.
    """
    livetime = 227708167.68
    energy_bins = np.logspace(4, 7, 3 * 20 + 1)
    energy_bins_new = np.logspace(4, 7, 28)
    
    # Get the effective area
    eff_df = get_effective_area_dataframe(energy_bins)

    # Get nuSIprop flux data
    flx_df = pd.read_csv(os.path.join(base_path, 'flux/flux_Fig6_600bins.csv'), index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    bin_centers = flx_df.index.values
    bin_edges = bin_centers_to_edges(bin_centers)
    delta_E = np.diff(bin_edges)

    total_events_df = total_events(flx=eff_df, eff=flx_df, norm=0.8*1e-13, livetime=livetime, delta_E=delta_E)
    
    # Get MC events and bin them equally to the nuSIprop flux bins
    mc, weights = get_weights(energy_bins, livetime=livetime, gen2=False)
    mc_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=bin_edges, weights=weights[0]) 
    mc_df = pd.DataFrame({
        'bin_centers': bin_centers,
        'events': mc_events,
    })
    mc_df.set_index('bin_centers', inplace=True)
    
    # Apply energy smearing
    total_events_df['with_resolution'] = apply_energy_smearing(
        energies=np.asarray(total_events_df.index), 
        events=np.asarray(total_events_df['total_events']),
        resolution=0.1
        )

    mc_df['with_resolution'] = apply_energy_smearing( 
        energies=np.asarray(mc_df.index), 
        events=np.asarray(mc_df['events']), 
        resolution=0.1)
    
    print(f"mc_df: {mc_df}")
    print(f"Total events: {total_events_df}")
    
    
    plt.hist(
        total_events_df.index.values, 
        weights=total_events_df['total_events'], 
        bins=energy_bins_new,
        label='no smearing',
        histtype='step',
        color='lightgray'
    )
    plt.hist(
        mc_df.index.values, 
        weights=mc_df['events'], 
        bins=energy_bins_new,
        label='MC events (no smearing)',
        histtype='step',
        color='lightgray'
    )
    
    
    # Plotting
    plt.hist(
        total_events_df.index.values,
        weights=total_events_df['with_resolution'], 
        bins=energy_bins_new, 
        label='Total Events', 
        histtype='step', 
        color='blue'
    )
    
    plt.hist(  
        mc_df.index.values,
        weights=mc_df['with_resolution'], 
        bins=energy_bins_new, 
        label='MC Events', 
        histtype='step', 
        color='orange'
    )
    
    plt.xscale('log')
    plt.xlim(6*energy_bins_new[0], energy_bins_new[-1])
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('Fig 6')
    plt.legend()
    plt.show()