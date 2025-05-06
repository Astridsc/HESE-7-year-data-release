import sys
import os
import os.path
import matplotlib
import matplotlib.style

matplotlib.use("TkAgg")
matplotlib.style.use("./resources/mpl/paper.mplstyle")
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import json
import pandas as pd

import data_loader
import weighter
import binning


#outdir = "./effective_areas/"

# Disable LaTeX text rendering
#plt.rcParams['text.usetex'] = False


mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]



def center(x):
    x = np.asarray(x)
    return (x[1:] + x[:-1]) / 2.0


def get_particle_masks(particleType):
    """
    Get a dictionary containing masks by particle type.
    """
    particle_dict = {
        "eminus": 11,
        "eplus": -11,
        "muminus": 13,
        "muplus": -13,
        "tauminus": 15,
        "tauplus": -15,
        "nue": 12,
        "nuebar": -12,
        "numu": 14,
        "numubar": -14,
        "nutau": 16,
        "nutaubar": -16,
    }
    abs_particle_dict = {
        "e": 11,
        "mu": 13,
        "tau": 15,
        "2nue": 12,
        "2numu": 14,
        "2nutau": 16,
    }
    other_particle_dict = {
        "nu": lambda x: (
            lambda xx: functools.reduce(
                np.logical_or, [(xx == 12), (xx == 14), (xx == 16)], np.zeros(xx.shape)
            )
        )(abs(np.array(x))),
        "all": lambda x: np.ones(np.array(x).shape).astype(bool),
    }
    masks = {}
    for name, id in particle_dict.items():
        mask = particleType == id
        if np.any(mask):
            masks[name] = mask
    for name, id in abs_particle_dict.items():
        mask = abs(particleType) == id
        if np.any(mask):
            masks[name] = mask
    for name, id in other_particle_dict.items():
        mask = id(particleType)
        if np.any(mask):
            masks[name] = mask
    return masks


def plot_effective_areas(json_files=mc_filenames):
    # Load the MC
    json_data = dict()
    for filename in json_files:
        json_data.update(json.load(open(filename, "r")))

    # Get the MC generation information
    weight_over_flux_over_livetime = np.array(json_data["weightOverFluxOverLivetime"])

    # Choose the energy binning
    energy_bins = np.logspace(2, 7, 5 * 20 + 1)  # 1e2 to 1e7 with 20 bins per decade
    energy_bin_widths = np.diff(energy_bins)

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

    ## Now we have what we need to compute the effective area ##

    # Choose the color map
    cm = plt.get_cmap("plasma")

    # Choose some line styles
    line_styles = ["-", "--", ":", ":", "-", "--", ":", ":"]

    # 3 flavors in the MC
    n_flavors = 3

    # We are going to average our effective area over the whole sky
    total_angular_width = 4.0 * np.pi

    bin_widths = energy_bin_widths * total_angular_width

    # A meter is 100cm
    meter = 100

    # How to compute and plot the effective area (in a histogram style with errors)
    def plot_line(ax, masks, color, line_style, label, factor=1.0):
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
        effective_area_m2 = effective_area_cm2 / (meter ** 2)
        effective_area_m2_error = effective_area_cm2_error / (meter ** 2)

        # Plot things only if they will appear on the plot
        if np.any(effective_area_m2 > 1e-4):
            # Make plot of effective area
            ax.step(
                energy_bins[1:],
                effective_area_m2,
                color=color,
                linestyle=line_style,
                lw=2,
                label=label,
            )
            # Add the errorbars to the plot
            ax.errorbar(
                10 ** center(np.log10(energy_bins)),
                effective_area_m2,
                yerr=effective_area_m2_error,
                color=color,
                linestyle="none",
            )
        return effective_area_m2, effective_area_m2_error
    

        # How to format the axis
    def format_axis(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim((1e4, 1e8))
        ax.set_ylim((1e-4, 2e3))
        ax.set_xlabel("Neutrino Energy [GeV]")
        ax.set_ylabel("Effective Area [m²]")

        # Override the yaxis tick settings
        major = 10.0 ** np.arange(-3, 5)
        minor = np.arange(2, 10) / 10.0
        locmaj = matplotlib.ticker.FixedLocator(10.0 ** np.arange(-2, 4))
        locmin = matplotlib.ticker.FixedLocator(
            np.tile(minor, len(major)) * np.repeat(major, len(minor))
        )
        locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,), numticks=12)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=minor, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)

        ax.legend(frameon=True, loc="upper left")


    # How to save the figure with nice spacing
    def save(fig, name):
        path = os.path.dirname(os.path.abspath(__file__))
        fig.tight_layout()
        fig.savefig(os.path.join(path, name))   
        #fig.savefig("~/HESE-7year-data-release/HESE-7-year-data-release" + name + ".png")
        fig.clf()
           

    # Let's make an effective area vs. energy plot split by neutrino flavor
    fig, ax = plt.subplots(figsize=(7, 5))
    eff, eff_err = [], []
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        line_style = line_styles[0]
        label = f"ν_{flavor} + ν̄_{flavor}"
        particle_key = "2nu" + flavor
        particle_mask = particle_masks[particle_key]
        masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
        # The factor of 0.5 is needed so that we compute the average
        # neutrino/antineutrino effective area. This is in contrast to the
        # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
        # sum of the neutrino and antineutrino effective areas.
        eff_f, eff_err_f = plot_line(ax, masks, color, line_style, label, factor=1)
        eff.append(eff_f)
        eff_err.append(eff_err_f)
    format_axis(ax)
    save(fig, "test")
    #print('effective area:  ', eff)
    return eff, eff_err

eff, eff_err = plot_effective_areas()
eff = [eff[0], eff[1], eff[2]]  # Dont want to distinguish between particle/antiparticle
for eff_ in eff:
    eff_ = [2*x for x in eff_]    #  Double the effective area to account for particle/antiparticle


# Function to adjust the effective area array for new energy limits
# given it is approximately linear in loglog space (Fig 25).
def get_effective_area(eff, emin, gen2=True):
    # Filter energy bins in the range [10^5, 10^8]
    energy_bins = np.logspace(2,7, 5*20+1)
    mask = (energy_bins >= emin) 
    energy_bins_filtered = energy_bins[mask]
    #energy_bins_filtered = energy_bins[(energy_bins >= emin)]

    # Extrapolate the area for higher energies if needed
    energy_bins_new_range = np.logspace(7, 8, num=20)
    energy_bins_combined = np.concatenate((energy_bins_filtered, energy_bins_new_range))

    m = 0.31  # Slope calculated 
    b = -0.55  # Intercept calculated 

    projected_eff = np.asarray(2*10**(m * np.log10(energy_bins_new_range) + b))    # At higher energies, the effective area is the same for all flavors

    eff_new = []
    for eff_ in eff:
        ni = len(energy_bins)
        nf = len(energy_bins_filtered)
        n_delete = ni - nf
        eff_ = np.delete(eff_, range(n_delete-1))
        eff_new.append(np.concatenate((eff_, projected_eff)))

    if gen2 == True:
        eff_new = [10*x for x in eff_new]       # To account for factor 10 as: A_eff(Gen2) ~ 10* A_eff(Current)

    return eff_new, energy_bins_combined


def rebinning(flx, eff, nbins):

    # Interpolate `flx` to the same energy bins as `eff`
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)

    # Compute total events as eff * flx
    total_events = eff * flx_interpolated
    total_events['total'] = total_events['nu_e'] + total_events['nu_mu'] + total_events['nu_tau']


    e_edges, _, _ = binning.get_bins(emin=100.0e3, emax=1e8, ewidth=0.1, eedge=100.0e3)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

    log_bins = np.logspace(np.log10(1e5), np.log10(1e8), nbins+1)  # log-spaced bins

    # Group data into logarithmic bins
    total_events_binned = total_events.groupby(pd.cut(total_events.index, bin_centers)).sum()

    # Compute the midpoint (center) of each logarithmic interval
    # Geometric mean for log step midpoints
    total_events_binned['interval_center'] = [
        (interval.left * interval.right) ** 0.5 for interval in total_events_binned.index
    ]
    print(total_events_binned)

    return total_events_binned, e_edges


def total_events(flux, effective_area, save_to_csv=True):
    total_events_df = flux * effective_area
    total_events_df['total'] = total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau']

    if save_to_csv==True:
        total_events_df.to_csv('total_events.csv')

    return total_events


def apply_energy_resolution(res, dataframe, events):
    backup = dataframe.copy()
    corrections = np.zeros(shape=np.shape(events))

    for index, events_ in enumerate(events):
        spill = res*events_
        if index > 0:
            corrections[index - 1] += 0.5*spill
        if index < len(events)-1:
            corrections[index + 1] += 0.5*spill          
        corrections[index] += - spill
    dataframe['with_resolution'] = events + corrections

    return dataframe


def plot_fig8():
    # Compute limited/extrapolated effective area and energy bins 
    eff_new, energy_bins_new = get_effective_area(eff, emin=1e5)
    eff_new = np.asarray(eff_new)

    # Set up pandas dataframes
    eff_df = pd.DataFrame(eff_new.T, index=energy_bins_new, columns=['nu_e', 'nu_mu', 'nu_tau'])
    eff_df.to_csv('effective_areas_by_flavor.csv')

    flx_df = pd.read_csv('flux_Fig8.csv', index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    # Compute the total events with the right amount of bins
    total_events_df, e_edges = rebinning(flx_df, eff_df, nbins=30)

    # Apply detector energy resolution to the histogram
    total_events_df = apply_energy_resolution(0.1, total_events_df, np.asarray(total_events_df['total']))
    print(total_events_df['with_resolution'])
    total_events_df.to_csv('Total_events.csv')

    N_tot = total_events_df['total']
    N_tot_res = total_events_df['with_resolution']
    energies = total_events_df['interval_center']

    fig, ax = plt.subplots()

    plt.step(energies, N_tot, label='Without resolution')
    plt.step(energies, N_tot_res, label='With resolution')

    plt.xlim(200.0e3, 1e8)
    plt.ylim(1.0e-2, 1.0e2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('IceCube-Gen2')
    plt.legend()
    plt.show()


def plot_fig6():
    # Compute limited/extrapolated effective area and energy bins 
    eff_new, energy_bins_new = get_effective_area(eff, emin=1e4, gen2=False)
    eff_new = np.asarray(eff_new)

    # Set up pandas dataframes
    eff_df = pd.DataFrame(eff.T, index=energy_bins_new, columns=['nu_e', 'nu_mu', 'nu_tau'])
    eff_df.to_csv('effective_areas_by_flavor.csv')

    flx_df = pd.read_csv('flux_Fig6.csv', index_col=0)
    flx_df.index = flx_df.index / 1e9    # Convert to [GeV]

    # Compute the total events with the right amount of bins
    total_events_df, e_edges = rebinning(flx_df, eff_df, nbins=30)

    # Apply detector energy resolution to the histogram
    total_events_df = apply_energy_resolution(0.1, total_events_df, np.asarray(total_events_df['total']))
    total_events_df.to_csv('Total_events.csv')

    livetime = 7.5*365*24*3600

    N_tot = total_events_df['total'] * livetime
    N_tot_res = total_events_df['with_resolution'] *livetime
    energies = total_events_df['interval_center']

    #fig, ax = plt.subplots()

    plt.step(energies, N_tot, label='Without resolution')
    plt.step(energies, N_tot_res, label='With resolution')

    plt.xlim(60.0e3, 1e7)
    #plt.ylim(1.0e-2, 1.0e2)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Number of events')
    plt.title('IceCube 7.5 years')
    plt.legend()
    plt.show()


plot_fig8()





# Customize the plot
#plt.plot(eff_df.index, eff_df['nu_e'], label='Effective area')
#plt.plot(flx_df.index, flx_df['nu_e'], label='Flux')
#plt.plot(total_events_df['interval_center'], total_events_df['nu_e'], label='Total events')
#fit, ax = plt.subplots(2)



"""
ax.hist(
    energies[:n],
    weights=N_tot[:n],
    bins=e_edges[:n],
    log=True,
    histtype="step",
    label='Without resolution',
    color='r',
)

ax.hist(
    energies[:n],
    weights=N_tot_res[:n],
    bins=e_edges[:n],
    log=True,
    histtype="step",
    label='With resolution',
    color='b',
)"""


# Specify the column you want to plot the histogram for
# For example, use the `total` column, as it represents the summed values for all ν types.
"""
plt.bar(eff_df.index, eff_df['nu_e'], width=2, label=r'$\nu_e$')
plt.bar(eff_df.index, eff_df['nu_tau'], width=2, label=r'$\nu_{\tau}$')
plt.bar(eff_df.index, eff_df['nu_mu'], width=2, label=r'$\nu_{\mu}$')"""
#plt.bar(total_events_df.index, total_events_df['total'], width=2, align='edge', label="Total Events")
#plt.step(eff_df.index, eff_df['nu_tau'])
"""eff_interpolated = pd.DataFrame(
    {col: interp1d(eff_df.index, eff_df[col], bounds_error=False, fill_value="extrapolate")(flx_df.index)
     for col in eff_df.columns},
    index=flx_df.index)
total_df = eff_interpolated * flx_df
total_df['total'] = total_df['nu_e'] + total_df['nu_mu'] + total_df['nu_tau']

plt.step(flx_df.index, 10* livetime*total_df['total'], label='Total')

# Show the histogram
plt.show()"""
    


    





    