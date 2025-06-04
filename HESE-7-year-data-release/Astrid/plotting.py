import sys
import os
import os.path
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import matplotlib
import matplotlib.style
matplotlib.use("TkAgg")
matplotlib.style.use("./resources/mpl/paper.mplstyle")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import numpy as np

import json
import pandas as pd

#import data_loader
#import weighter
import binning

from Astrid.config import MC_FILENAMES
from Astrid.config import LIVETIME1, LIVETIME2, LIVETIME3
from Astrid.config import FLUX_FILE_6, FLUX_FILE_8, MC_GEN1_FILE, MC_GEN2_FILE
#from Astrid.config import LIVETIME1, LIVETIME2, LIVETIME3, FLUX_FILE, MC_FILENAMES 
from Astrid.data_processing import load_true_events, get_particle_masks, get_weights, load_flux_data
from Astrid.effective_area import get_effective_area_range, rebinning, total_events, bin_weights, rebinning_old
from Astrid.effective_area import apply_energy_smearing, HESE_effective_areas, get_effective_area_dataframe

#outdir = "./effective_areas/"

# Disable LaTeX text rendering
#plt.rcParams['text.usetex'] = False

# filepath: \\wsl.localhost\Ubuntu\home\astridaurora\HESE-7-year-data-release\HESE-7-year-data-release\Astrid\plotting.py


def bin_centers_to_edges(bin_centers):
    # Assumes log-spaced bins
    log_centers = np.log10(bin_centers)
    dlog = np.diff(log_centers)
    # Internal edges are halfway between centers in log-space
    log_edges = np.zeros(len(bin_centers) + 1)
    log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
    # Extrapolate first and last edge
    log_edges[0] = log_centers[0] - dlog[0]/2
    log_edges[-1] = log_centers[-1] + dlog[-1]/2
    return 10**log_edges


def test_smearing():

    energies = np.logspace(4, 7, 100)
    events = np.zeros_like(energies)
    events[50] = 1000  # All events in one bin

    smeared = apply_energy_smearing(energies, events, resolution=0.1)

    plt.step(energies, events, where='mid', label='Original')
    plt.step(energies, smeared, where='mid', label='Smeared')
    plt.xscale('log')
    plt.legend()
    plt.show()



def plot_fig6():
    # Expected number of events with/without nuSI.
    # Assuming g=0.1, Mphi=5MeV, \Sum(m_nu)=0.1eV, Livetime=7.5
    # Adjust normalisation (and exposure?..) to observe 10 events in lowest bin (at 60TeV)

    norm = (1/1.2)*1e-12
    livetime = 227708167.68 # Used by HESE taking into account some breaks in the runtime

    Edep = np.logspace(4, 7, num=3*20+1)
    eff_df = get_effective_area_dataframe(Edep, gen2=False)
    print('eff_df: ', eff_df)

    # Get flux from nuSIprop
    flx_df = pd.read_csv('Astrid/flux/flux_Fig6.csv', index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    delta_E = np.diff(flx_df.index.values)
    delta_E = np.append(delta_E, delta_E[-1])  # Append last value to match length

    # Get total events from total_events = flx_interpolated * eff
    total_events_df = total_events(eff=flx_df, flx=eff_df, livetime=livetime, norm=norm, delta_E=delta_E, save_to_csv=False)

    # Apply energy smearing
    res = 0.1  # Detector resolution of 10%
    smeared_events = apply_energy_smearing(
        energies=np.asarray(total_events_df.index), 
        events=np.asarray(total_events_df['total_events']),
        resolution=res
        )

    total_events_df['with_resolution'] = smeared_events
    print('total_events_df: ', total_events_df)

    # Compute the total events with the right amount of bins
    Edep_new_binning = np.logspace(4, 7, num=27)
    total_events_df = rebinning(total_events_df, Edep_new_binning)

    N_tot = total_events_df['total_events']
    N_tot_res = total_events_df['with_resolution']
    energies = total_events_df['interval_center'] 

    bin_centers = energies.values
    e_edges = bin_centers_to_edges(bin_centers=bin_centers)

    # Get the HESE MC events for given energy range & bins
    mc, weights = get_weights(Edep_new_binning, livetime=livetime, gen2=False)
    df_binned = bin_weights(mc, weights[0], e_edges)

    #plt.step(df_binned['edges'], df_binned['sum_weights'], label='HESE mc, ', color='b')

    plt.step(energies, N_tot, label='Without resolution', color='lightgrey')
    plt.step(energies, N_tot_res, label='With resolution' + r'$\gamma$= 2.0' + ', ' + r'$R$= ' + str(res), color='purple')

    plt.hist(
        [mc["recoDepositedEnergy"]],
        weights=weights,
        bins=bin_centers,
        histtype="step",
        stacked=True,
        label='No ' + r'$\nu SI$',
        color='green',
        linestyle='dashed',
    )

    #plt.step(energies, weights, label='HESE mc', color='b')

    plt.xlim(6*Edep[0], Edep[-1])
    #plt.ylim(0, 14)
    #plt.ylim(1.0e-2, 1.0e2)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('IceCube 7.5 years, g=0.1, ' + r'$M_{\phi}=5MeV$' + ', ' + r'$\sum m_{\nu}=0.1eV$')
    plt.legend()
    plt.show()



def plot_fig8():

    norm = 1e-10    # Lite oklart med normalisering?..
    Edep = np.logspace(5, 8, num=3*20+1)

    # Get the effective area as originally provided by HESE (1e4 to 1e7), [m2]
    eff, eff_err = plot_effective_areas() 

    # Compute limited/extrapolated effective area and energy bins 
    eff_new, energy_bins_new = get_effective_area_range(eff, Edep, gen2=True)
    eff_new = np.asarray(eff_new)

    # Set up pandas dataframes
    eff_df = pd.DataFrame(eff_new.T, index=energy_bins_new, columns=['nu_e', 'nu_mu', 'nu_tau'])
    eff_df.to_csv('effective_areas_by_flavor.csv')

    flx_df = pd.read_csv('Astrid/flux/flux_Fig8.csv', index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    # Compute the total events with the right amount of bins
    #total_events_df, e_edges = rebinning(flx_df, eff_df, nbins=30)
    #total_events_df = pd.read_csv('Total_events.csv')
    #total_events_df = total_events(flx_df, eff_df, save_to_csv=False)
    total_events_df = total_events(flx_df, eff_df, save_to_csv=False)
    print('len of total events flx interpolated as eff ', len(total_events_df))

    energies = np.asarray(total_events_df.index)
    events = np.asarray(total_events_df['total'])
    resolution = 0.1  # Detector resolution of 10%

    # Apply energy smearing
    smeared_events = apply_energy_smearing(energies, events, resolution)
    total_events_df['with_resolution'] = smeared_events
    #print(total_events_df)

    # Compute the total events with the right amount of bins
    Edep = np.logspace(5, 8, num=30)
    total_events_df = rebinning(total_events_df, Edep)
    #total_events_df.to_csv('Total_events_Fig8.csv')

    N_tot = total_events_df['total']
    N_tot_res = total_events_df['with_resolution']
    energies = total_events_df['interval_center']

    #fig, ax = plt.subplots()

    plt.step(energies, norm* LIVETIME2 *N_tot_res, label='With resolution')
    plt.step(energies, norm*LIVETIME2*N_tot, label='Without resolution')
    #plt.step(energies, eff_df['nu_e'], label='Effective area [m2], nu_e', color='g')


    plt.xlim(200.0e3, 1e8)
    #plt.ylim(1.0e-2, 1.0e2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('IceCube-Gen2')
    plt.legend()
    plt.show()



def plot_Fig8():
    norm = (1/5) * 1e-12    # Lite oklart med normalisering?..

    # Get the effective area as originally provided by HESE (1e4 to 1e7), [m2]
    Edep = np.logspace(5, 8, num=3*20+1)
    eff_df = get_effective_area_dataframe(Edep, gen2=True)

    flx_df = pd.read_csv('Astrid/flux/flux_Fig8_si2.csv', index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    delta_E = np.diff(flx_df.index.values)
    delta_E = np.append(delta_E, delta_E[-1])  # Append last value to match length
    # Convert bin centers to edges
    #bin_edges = bin_centers_to_edges(flx_df.index.values)
    #delta_E = np.diff(bin_edges)
    #print(np.diff(flx_df.index.values))
    #print('delta_E: ', delta_E)

    # Compute the total events
    total_events_df = total_events(eff=flx_df, flx=eff_df, livetime=LIVETIME2, norm=norm, delta_E=delta_E, save_to_csv=False)
    print('total_events_df: ', total_events_df)
    print('length of total events: ', len(total_events_df))

    # Apply Gaussian energy smearing with 10% resolution
    res=0.1
    smeared_events = apply_energy_smearing( 
        energies=np.asarray(total_events_df.index), 
        events=np.asarray(total_events_df['total_events']), 
        resolution=res)
    total_events_df['with_resolution'] = smeared_events

    # Compute the total events with the right amount of bins
    Edep_new = np.logspace(5, 8, num=31)
    #E_edges = bin_centers_to_edges(bin_centers=Edep_new)
    #total_events_df = rebinning_old(total_events_df, Edep_new)
    total_events_df = rebinning(total_events_df, Edep_new)
    #energies = np.asarray(total_events_df['interval_center'])
    energies = np.asarray(total_events_df.index)
    print('total_events_df: ', total_events_df)

    # Getting HESE MC events for given energy range & bins

    #e_edges = bin_centers_to_edges(bin_centers=energies)
    mc, weights = get_weights(Edep_new, livetime=LIVETIME2, gen2=True)
    #mc_binned = bin_weights(mc, weights[0], e_edges=e_edges)
    #print(len(bin_centers), len(e_edges))
    
    mc_df = pd.DataFrame({
        'recoDepositedEnergy': mc['recoDepositedEnergy'],
        'weights': np.asarray(weights[0]),
    })
    mc_df.set_index('recoDepositedEnergy', inplace=True)
    print('len of mc_df: ', len(mc_df))
    print('mc_df: ', mc_df)
    mc_df = rebinning(mc_df, Edep_new)


    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 9),
                            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.step(energies, total_events_df['total_events'], label='Without resolution', color='lightgrey', linestyle='dashed')
    ax1.step(energies, total_events_df['with_resolution'], label=r'$\gamma$=2.0 '+', R='+str(res), color='r')
    ax1.step(Edep_new[1:], mc_df['weights'], label='HESE mc', color='b')

    cm = plt.get_cmap("magma")
    ax1.hist(
        [mc["recoDepositedEnergy"]],
        weights=weights,
        bins=energies,
        histtype="bar",
        stacked=True,
        label='No ' + r'$\nu SI$, ' + r'$\gamma$= 2.9',
        color=cm(0.75),
    )

    # Plotting and computing histogram for the detector sensitivity.
    # Compute histograms for both distributions with the SAME bins
    mc_hist, _ = np.histogram(mc["recoDepositedEnergy"], bins=energies, weights=weights[0])
    flx_hist, _ = np.histogram(energies, bins=energies, weights=total_events_df['with_resolution'])

    # Remove zero values in `flx_hist` to avoid division by zero during normalization
    nonzero = flx_hist > 0
    mc_hist = mc_hist[nonzero]
    flx_hist = flx_hist[nonzero]

    # Compute normalized difference
    normalized_diff = (flx_hist - mc_hist) / np.sqrt(flx_hist)

    ax2.hist(
        energies[:-1][nonzero],  # Bin centers corresponding to non-zero elements
        bins=energies,
        weights=normalized_diff,
        histtype="step",
        label=r"Normalized Difference",
        color="b",
    )

    ax1.loglog()
    plt.xlim(2*Edep[0], Edep[-1])
    ax1.set_ylim(4*1.0e-1, 1.0e2)
    ax1.set_ylabel("Number of events")
    ax1.yaxis.set_ticks_position('both')      # Show ticks on both left and right
    ax1.tick_params(axis='y', which='both', right=True, labelright=False)  # Enable right ticks, hide right labels
    ax1.grid(True, which='both', axis='both', alpha=0.3, color='lightgrey')
    ax1.set_title("IceCube Gen2, 10 years livetime")
    ax1.legend()
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel(r'$\Delta N/\sqrt{\Delta N_{\nu SI}}$')
    ax2.set_xlabel(r"$E_{dep} [GeV]$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()


def test_binning():
    df = pd.DataFrame({
        'col1': np.logspace(4, 7, num=300),
        'col2': np.ones(300),  # Example data
    })
    df.set_index('col1', inplace=True)
    print('Original DataFrame:')
    print(df)
    E_dep = np.logspace(4, 7, num=32)
    df_binned = rebinning(df, E_dep)
    print('Binned DataFrame:')
    print(df_binned)


def test_rebinning_old():
    df = pd.DataFrame({
        'col1': np.logspace(4, 7, num=300),
        'col2': np.ones(300),  # Example data
    })
    df.set_index('col1', inplace=True)
    print('Original DataFrame:')
    print(df)
    E_dep = np.logspace(4, 7, num=32)
    df_binned = rebinning_old(df, E_dep)
    print('Binned DataFrame:')
    print(df_binned)


def mockdata():
    """Edep1 = np.logspace(4, 5, num=20)
    Edep2 = np.logspace(5, 7, num=40)

    # Get the effective area as originally provided by HESE (1e4 to 1e7), [m2]
    # Compute limited/extrapolated effective area and energy bins 
    eff_df_1 = get_effective_area_dataframe(Edep1, gen2=False)
    eff_df_2 = get_effective_area_dataframe(Edep2, gen2=True)
    eff_df_1.to_csv('effective_areas_by_flavor_gen1.csv')
    eff_df_2.to_csv('effective_areas_by_flavor_gen2.csv')

    bin_centers1 = eff_df_1.index.values
    bin_centers2 = eff_df_2.index.values

    e_edges1 = bin_centers_to_edges(bin_centers1)
    e_edges2 = bin_centers_to_edges(bin_centers2)

    mc1, weights1 = get_weights(e_edges1, livetime=LIVETIME3, gen2=False)
    mc2, weights2 = get_weights(e_edges2, livetime=LIVETIME2, gen2=True)

    mc_events_binned1 = bin_weights(mc1, weights1[0], e_edges1, bin_centers1)
    mc_events_binned2 = bin_weights(mc2, weights2[0], e_edges2, bin_centers2)

    mc_events_binned1.to_csv('mc_Gen1.csv')
    mc_events_binned2.to_csv('mc_Gen2.csv')
    print(len(mc_events_binned1))
    print(len(mc_events_binned2))
    #plt.step(mc_events_binned1['bin_centers'], mc_events_binned1['sum_weights'], label='HESE mc1', color='b')
    #plt.step(mc_events_binned2['bin_centers'], mc_events_binned2['sum_weights'], label='HESE mc2', color='r')
    #plt.step(total_events_binned1['Bin_Center'], total_events_binned1['Total_Events'], color='g', label='Rebinned by total events')
    #plt.step(total_events_binned2['Bin_Center'], total_events_binned2['Total_Events'], color='r', label='Rebinned by total events')
    
    #plt.step(eff_df_1.index, eff_df_1['nu_e'], label='Aeff [m2], nu_e', color='g')
    #plt.step(eff_df_2.index, eff_df_2['nu_e'], label='Aeff [m2], nu_e', color='orange')
    plt.step()
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()"""


    Edep1 = np.logspace(4, 5, num=20+1)
    Edep2 = np.logspace(5, 7, num=40+1)
    print(len(Edep1), len(Edep2))


    # Get the effective area as originally provided by HESE (1e4 to 1e7), [m2]
    # Compute limited/extrapolated effective area and energy bins 
    eff1 = get_effective_area_dataframe(Edep1, gen2=False)
    eff2 = get_effective_area_dataframe(Edep2, gen2=True)
    eff1.to_csv('effective_areas_by_flavor_gen1.csv')
    eff2.to_csv('effective_areas_by_flavor_gen2.csv')
    energies1 = np.asarray(eff1.index)
    energies2 = np.asarray(eff2.index)

    mc1, weights1 = get_weights(Edep1, livetime=LIVETIME3, gen2=False)
    mc2, weights2 = get_weights(Edep2, livetime=LIVETIME2, gen2=True)
    mc_df1 = pd.DataFrame({
            'Edep': mc1['recoDepositedEnergy'],
            'weights': np.asarray(weights1[0]),
        })
    mc_df2 = pd.DataFrame({
            'Edep': mc2['recoDepositedEnergy'],
            'weights': np.asarray(weights2[0]),
        })
    mc_df1.set_index('Edep', inplace=True)
    mc_df2.set_index('Edep', inplace=True)
    mc_events_binned1 = rebinning(mc_df1, Edep1)
    mc_events_binned2 = rebinning(mc_df2, Edep2)
    
    print('mc_df1: ', mc_events_binned1)
    print('mc_df2: ', mc_events_binned2)
    mc_events_binned1.to_csv('mc_Gen1.csv')
    mc_events_binned2.to_csv('mc_Gen2.csv')
    print(len(mc_events_binned1), len(eff1))
    print(len(mc_events_binned2), len(eff2))




def Fig10():
    # Mock Data, 15 years exposure to Gen1
    #livetime1 = 15*365*24*3600
    #norm = (4/3)*1e12

    Edep1 = np.logspace(4, 5, num=20)
    mc1, weights1, e_edges1, bin_centers1, colors1 = get_weights(Edep1, livetime=LIVETIME3, gen2=False)
    #print(len(weights1[0]), weights1[0])
    #print(len(mc1['recoDepositedEnergy']), mc1['recoDepositedEnergy'])
    mc1_dict = {'E_dep': mc1['recoDepositedEnergy'], 'Weights': weights1[0]}
    mc1_df = pd.DataFrame(data=mc1_dict)
    mc1_df.to_csv('mc_Gen1.csv')


    # Mock Data, 10 years exposure to Gen2
    #livetime2 = 10*365*24*3600
    Edep2 = np.logspace(5, 8, num=3*len(Edep1))
    #print('Edep1', Edep1)
    #print('Edep2', Edep2)

    mc2, weights2, e_edges2, bin_centers2, colors2 = get_weights(Edep2, livetime=LIVETIME2, gen2=True)
    #print(len(weights2[0]), weights2[0])
    #print(len(mc2['recoDepositedEnergy']), mc2['recoDepositedEnergy'])
    mc2_dict = {'E_dep': mc2['recoDepositedEnergy'], 'Weights': weights2[0]}
    mc2_df = pd.DataFrame(data=mc2_dict)
    #mc2_df = mc2_df.groupby(pd.cut(total_events.index, bin_centers)).sum()

    # Now plotting the histogram using plt.hist for comparison

    """plt.hist(
        mc1['recoDepositedEnergy'],
        weights=weights1[0],
        bins=e_edges1,
        histtype="step",
        label='Gen1, ' + str(len(bin_centers)) + ' bins'
    )"""

    plt.hist(
    len(weights1) * [mc1["recoDepositedEnergy"]],
    weights=weights1,
    bins=e_edges1,
    histtype="step",
    stacked=True,
    label='Gen1, ' + str(len(Edep1)) + ' bins'
    )


    # Bin the data using pd.cut
    mc1_df['Bin'] = pd.cut(mc1_df['E_dep'], bins=e_edges1, labels=bin_centers1)

    # Group by bins and sum the weights
    # Rename the bin column for clarity
    total_events_binned1 = mc1_df.groupby('Bin')['Weights'].sum().reset_index()
    total_events_binned1.rename(columns={'Bin': 'Bin_Center', 'Weights': 'Total_Events'}, inplace=True)

    mc2_df['Bin'] = pd.cut(mc2_df['E_dep'], bins=e_edges2, labels=bin_centers2)

    # Group by bins and sum the weights
    total_events_binned2 = mc2_df.groupby('Bin')['Weights'].sum().reset_index()

    # Rename the bin column for clarity
    total_events_binned2.rename(columns={'Bin': 'Bin_Center', 'Weights': 'Total_Events'}, inplace=True)


    """Aeff1 = pd.read_csv('Aeff_mockdata1.csv')
    Aeff2 = pd.read_csv('Aeff_mockdata2.csv')

    Aeff_binned1 = pd.DataFrame(
    {'Total_events': interp1d(mc1_df['Edep'], mc1_df['Weights'], bounds_error=False, fill_value="extrapolate")(eff1.index)},
    index=eff1.index)

    Aeff_binned2 = pd.DataFrame(
    {'Total_events': interp1d(mc2_df['Edep'], mc2_df['Weights'], bounds_error=False, fill_value="extrapolate")(eff2.index)},
    index=eff2.index)"""

    # Result: A DataFrame with bin centers and total events in each bin
    print(total_events_binned1)
    print(total_events_binned2)
    total_events_binned1.to_csv('mc_Gen1.csv')
    total_events_binned2.to_csv('mc_Gen2.csv')
    plt.step(total_events_binned1['Bin_Center'], total_events_binned1['Total_Events'], color='g', label='Rebinned by total events')
    plt.step(total_events_binned2['Bin_Center'], total_events_binned2['Total_Events'], color='r', label='Rebinned by total events')


    #eff_df = pd.DataFrame(eff_new.T, index=energy_bins_new, columns=['nu_e', 'nu_mu', 'nu_tau'])
    #eff_df.to_csv('effective_areas_by_flavor.csv')

    plt.hist(
        len(weights2) * [mc2["recoDepositedEnergy"]],
        weights=weights2,
        bins=e_edges2,
        histtype="step",
        stacked=True,
        label='Gen2, ' + str(len(Edep2)) + ' bins'
    )

    
    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(Edep1[0], Edep2[-1])

    plt.xlabel('Deposited Energy')
    plt.ylabel('Weighted Events')


    plt.legend()
    plt.show()


