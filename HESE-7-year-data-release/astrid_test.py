import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import matplotlib.pyplot as plt
import matplotlib
#import matplotlib.style
#matplotlib.style.use("./resources/mpl/paper.mplstyle")
#from matplotlib.font_manager import FontProperties
#import matplotlib.colors as mcolors

import scipy.stats
from scipy.interpolate import interp1d
import numpy as np

import data_loader
import weighter
import binning
import fc

import pandas as pd
import json
from tqdm import tqdm


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
]

# We plot using the best fit parameters found by HESE_fit.py
params = np.array(
    [
        -0.05309302,
        0.99815326,
        1.000683,
        2.0000000,
        6.36488608,
        1.00621679,
        0.95192328,
        -0.0548763,
        1.18706341,
        1.00013744,
        0.0,
    ]
)


livetime = 227708167.68
#livetime = 315360000
emin_ = 100.0e3
emax_ = 1e8

"""def load_true_events(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data"""

# Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]

def get_weights(Edep, livetime, gen2=False):
    print(Edep[0], Edep[-1])
    mc = data_loader.load_mc(mc_filenames, emin=Edep[0], emax=Edep[-1])
    data = data_loader.load_data("./resources/data/HESE_data.json", emin=Edep[0], emax=Edep[-1])

    #width = (np.log10(emax_) - np.log10(emin_)) / nbins
    #e_edges, _, _ = binning.get_bins(emin=emin_, emax=emax_, ewidth=width, eedge=emin_)
    print(len(mc))

    width = (np.log10(Edep[-1]) - np.log10(Edep[0])) / (len(Edep))
    e_edges, _, _ = binning.get_bins(emin=Edep[0], emax=Edep[-1], ewidth=width, eedge=Edep[0])
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))
    print(len(bin_centers))
    #n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

    weight_maker = weighter.Weighter(mc)
    #print(len(weight_maker))

    component_order = [
        ("astro_norm", "Astro."),
    ]

    params_dict = dict(zip(parameter_names, params))
    params_zeroed = params_dict.copy()

    for (zeroed_norm, _) in component_order:
        params_zeroed[zeroed_norm] = 0.0

    # We want to separate the histogram by components, so we separately get the weights
    # where the all normalization parameters but one are set to zero
    weights = []
    colors = []
    labels = []
    cm = plt.get_cmap("magma")
    color_scale = [cm(x) for x in [0.75]]
    for i, (zeroed_norm, zeroed_label) in enumerate(component_order):
        if params_dict[zeroed_norm] == 0.0:
            continue
        p_copy = params_zeroed.copy()
        p_copy[zeroed_norm] = params_dict[zeroed_norm]
        weights.append(
            weight_maker.get_weights(livetime, p_copy.keys(), p_copy.values())[0]
        )
        colors.append(color_scale[i])
        labels.append(zeroed_label)

    #Factor 10 to account for increase in effective area, which is approximately 10, according to Fig.25, Ref. 49
    if gen2==True:
        A_factor = 10
    else:
        A_factor = 1
    weights = [A_factor * weights[x] for x in range(len(weights))]
    
    return mc, weights, e_edges, bin_centers, colors


def plot_fig6():
    livetime = 7.5*365*24*3600
    norm = (4/3)*1e12
    #livetime = 227708167.68
    #livetime = 315360000
    #emin_ = 1e4
    #emax_ = 1e7
    Edep = np.logspace(4, 7, num=27)
    mc, weights, e_edges, bin_centers, colors = get_weights(Edep, livetime=livetime, gen2=False)
    print(weights)

    total_events_df = pd.read_csv('Astrid/total_events/Total_events_Fig6.csv')

    N_tot = total_events_df['total'] * livetime / norm
    N_tot_res = total_events_df['with_resolution'] * livetime / norm
    energies = total_events_df['interval_center']

    plt.step(energies, N_tot, label='Without resolution', color='grey')
    plt.step(energies, N_tot_res, label='With resolution', color='darkviolet')

    plt.hist(
        len(weights) * [mc["recoDepositedEnergy"]],
        weights=weights,
        bins=e_edges,
        histtype="step",
        stacked=True,
        label='No ' + r'$\nu SI$',
        color='g',
    )

    #plt.step(energies, weights, label='HESE mc')

    plt.xlim(6*Edep[0], Edep[-1])
    #plt.ylim(1.0e-2, 1.0e2)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.ylim([0, 16])
    plt.title('IceCube 7.5 years, g=0.1, ' + r'$M_{\phi}=5MeV$' + ', ' + r'$\sum m_{\nu}=0.1eV$')
    plt.legend()
    plt.show()


plot_fig6()



def plot_Fig8():
    livetime = 10*365*24*3600
    norm = 1e-12    # Lite oklart med normalisering?..

    emin_ = 1e5
    emax_ = 1e8

    # Plotting HESE MC events
    mc, weights, e_edges, colors = get_weights(emin_=emin_, emax_=emax_, livetime=livetime, nbins=30)
    
    # Plotting expected events as from nuSIprop
    nuSI_df = pd.read_csv('Total_events_Fig8.csv', index_col=0)

    N_nuSI = nuSI_df['with_resolution'] * livetime * norm
    energies = nuSI_df['interval_center']
    print(N_nuSI)


    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 9),
                        gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.hist(
        len(weights) * [mc["recoDepositedEnergy"]],
        weights=weights,
        bins=e_edges,
        histtype="bar",
        stacked=True,
        label='No ' + r'$\nu SI$',
        color=colors,
    )


    ax1.step(energies, N_nuSI, label='With resolution', color='r')

    ax1.loglog()
    plt.xlim(2*emin_, emax_)
    ax1.set_ylim(1.0e-1, 1.0e2)
    ax1.set_ylabel("Events over 10 years")

    # Simply reverses the order of the legend labels.
    #handles, labels = ax1.get_legend_handles_labels()
    #ax1.legend(handles[::-1], labels[::-1])
    #ax1.tight_layout()

    # Plotting and computing histogram for the detector sensitivity.
    # Compute histograms for both distributions with the SAME bins
    print('weights', len(weights), weights)
    print('mc recoDepositedEnergy', len(mc["recoDepositedEnergy"]), mc["recoDepositedEnergy"])

    mc_hist, _ = np.histogram(mc["recoDepositedEnergy"], bins=e_edges, weights=weights[0])
    flx_hist, _ = np.histogram(energies, bins=e_edges, weights=N_nuSI)

    # Remove zero values in `flx_hist` to avoid division by zero during normalization
    nonzero = flx_hist > 0
    mc_hist = mc_hist[nonzero]
    flx_hist = flx_hist[nonzero]

    # Compute normalized difference
    normalized_diff = (mc_hist - flx_hist) / np.sqrt(flx_hist)

    ax2.hist(
        e_edges[:-1][nonzero],  # Bin centers corresponding to non-zero elements
        bins=e_edges,
        weights=normalized_diff,
        histtype="step",
        label=r"Normalized Difference",
        color="b",
    )

    ax2.set_ylim(-5, 5)
    ax2.set_ylabel(r'$\Delta N/\sqrt{\Delta N_{\nu SI}}$')
    ax2.set_xlabel(r"$E_{dep} [GeV]$")

    plt.show()



def Fig10():
    # Mock Data, 15 years exposure to Gen1
    livetime1 = 15*365*24*3600
    #norm = (4/3)*1e12


    Edep1 = np.logspace(4, 5, num=20)
    mc1, weights1, e_edges1, bin_centers1, colors1 = get_weights(Edep1, livetime=livetime1, gen2=False)
    #print(len(weights1[0]), weights1[0])
    #print(len(mc1['recoDepositedEnergy']), mc1['recoDepositedEnergy'])
    mc1_dict = {'E_dep': mc1['recoDepositedEnergy'], 'Weights': weights1[0]}
    mc1_df = pd.DataFrame(data=mc1_dict)
    mc1_df.to_csv('mc_Gen1.csv')


    # Mock Data, 10 years exposure to Gen2
    livetime2 = 10*365*24*3600
    Edep2 = np.logspace(5, 8, num=3*len(Edep1))
    #print('Edep1', Edep1)
    #print('Edep2', Edep2)

    mc2, weights2, e_edges2, bin_centers2, colors2 = get_weights(Edep2, livetime=livetime2, gen2=True)
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



#Fig10()
