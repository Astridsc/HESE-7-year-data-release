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
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

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
        2.87375956,
        6.36488608,
        1.00621679,
        0.95192328,
        -0.0548763,
        1.18706341,
        1.00013744,
        0.0,
    ]
)



def load_true_events(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


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

"""
def get_HESE_mc_events(emin_, emax_, livetime, gen2=True):
    
    # Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.

    mc = data_loader.load_mc(mc_filenames, emin=emin_, emax=emax_)
    data = data_loader.load_data("./resources/data/HESE_data.json", emin=emin_, emax=emax_)

    e_edges, _, _ = binning.get_bins(emin=emin_, emax=emax_, ewidth=0.1, eedge=emin_)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

    # Oklart vad det här gör?
    #n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

    weight_maker = weighter.Weighter(mc)

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

    if gen2 == True:
        A_factor = 10
    else:
        A_factor = 1

    #Factor 10 to account for increase in effective area, which is approximately 10, according to Fig.25, Ref. 49
    weights = [A_factor * weights[x] for x in range(len(weights))]

    return mc, weights, e_edges, bin_centers"""



# Get MC weights for the astrophysical flux
def get_weights(emin_, emax_, livetime):
    mc = data_loader.load_mc(mc_filenames, emin=emin_, emax=emax_)
    print(len(mc))
    data = data_loader.load_data("./resources/data/HESE_data.json", emin=emin_, emax=emax_)

    e_edges, _, _ = binning.get_bins(emin=emin_, emax=emax_, ewidth=0.1, eedge=emin_)
    print(e_edges)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

    #n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

    weight_maker = weighter.Weighter(mc)
    print(weight_maker)


    component_order = [
        ("astro_norm", "Astro."),
    ]

    # Extract the single element from component_order
    zeroed_norm, zeroed_label = component_order[0]

    # Prepare parameters
    params_dict = dict(zip(parameter_names, params))
    params_zeroed = params_dict.copy()
    params_zeroed[zeroed_norm] = 0.0  # Zero out the normalization parameter

    # Only process if the parameter isn't already zero
    weights = []
    colors = []
    labels = []

    if params_dict[zeroed_norm] != 0.0:
        # Create a copy of params with only the selected normalization parameter active
        params_active = params_zeroed.copy()
        params_active[zeroed_norm] = params_dict[zeroed_norm]

        # Get the weights for the active parameter
        weight = weight_maker.get_weights(livetime, params_active.keys(), params_active.values())[0]
        weights.append(weight)

        # Set the color and label
        cm = plt.get_cmap("magma")
        color_scale = [cm(0.75)]  # Magma colormap, scaled at 0.75
        colors.append(color_scale[0])
        labels.append(zeroed_label)

    # Factor 10 to account for increase in effective area
    weights = [w for w in weights]

    """
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
    weights = [weights[x] for x in range(len(weights))]"""
    
    return mc, weights, e_edges, bin_centers, colors




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
    #eff, eff_err = plot_effective_areas()
    eff = [eff[0], eff[1], eff[2]]  # Dont want to distinguish between particle/antiparticle
    for eff_ in eff:
        eff_ = [2*x for x in eff_]    #  Double the effective area to account for particle/antiparticle
    
    # Oklart hur göra med eff_err
    return eff, eff_err


# Function to adjust the effective area array for new energy limits
# given it is approximately linear in loglog space (Fig 25).
"""def get_effective_area_range(eff, emin, Edep, gen2=True):
    emin, emax = Edep[0], Edep[-1]

    # Energy bins as originally provided by HESE
    energy_bins = np.logspace(2,7, 5*20+1)
    #mask = (energy_bins >= emin) 
    #energy_bins_filtered = energy_bins[mask]

    # Filter away 
    if Edep[-1] < energy_bins[-1]:
        mask = (energy_bins >= Edep[0] & energy_bins <= Edep[-1])
    else: 
        mask = (energy_bins >= Edep[0]) 
    energy_bins_filtered = energy_bins[mask]
    #energy_bins_filtered = energy_bins[(energy_bins >= emin)]

    # Extrapolate the area for higher energies if needed
    energy_bins_new_range = np.logspace(7, 8, num=20)

    m = 0.31  # Slope calculated 
    b = -0.55  # Intercept calculated 
    projected_eff = np.asarray(1e4 * 2*10**(m * np.log10(energy_bins_new_range) + b))    # At higher energies, the effective area is the same for all flavors

    eff_new = []
    for eff_ in eff:
        ni = len(energy_bins)
        nf = len(energy_bins_filtered)
        n_delete = ni - nf
        eff_ = np.delete(eff_, range(n_delete-1))
        if gen2 == True:
            eff_new.append(np.concatenate((eff_, projected_eff)))
        else:
            eff_new.append(eff_)

    if gen2 == True:
        eff_new = [10*x for x in eff_new]       # To account for factor 10 as: A_eff(Gen2) ~ 10* A_eff(Current)
        energy_bins_combined = np.concatenate((energy_bins_filtered, energy_bins_new_range))
        energy_bins = energy_bins_combined
    else:
        energy_bins = energy_bins_filtered

    return eff_new, energy_bins
"""


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
    energy_bins = np.logspace(2, 7, 5 * 20) 

    emin, emax = Edep[0], Edep[-1]

    # Filter energy bins within the range of Edep
    mask = (energy_bins >= emin) & (energy_bins <= min(emax, 1e7))
    energy_filtered = energy_bins[mask]

    # Extrapolate for Edep[-1] > 1e7
    if emax > 1e7:
        delta_e = np.log10(emax) - np.log10(1e7)
        print('delta_e: ', delta_e)
        num_bins = int(delta_e) * 20  # Number of bins to extrapolate
        print('num_bins: ', num_bins)
        # Combine filtered and extrapolated energy bins
        energy_extrapolated = np.logspace(7, np.log10(emax), num=num_bins) 
        energy_combined = np.concatenate((energy_filtered, energy_extrapolated))

        # Combine the filtered area and projected area for each flavor
        # Kolla upp faktor 1e4 * 2??????????!!
        projected_eff = (10 ** (0.31 * np.log10(energy_extrapolated) - 0.55))
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



def total_events(flx, eff, save_to_csv=True):
    # Interpolate `flx` to the same energy bins as `eff`
    print('flx: ', flx)
    flx_interpolated = pd.DataFrame(
    {col: interp1d(flx.index, flx[col], bounds_error=False, fill_value="extrapolate")(eff.index)
     for col in flx.columns},
    index=eff.index)
    print('flx interpolated: ', flx_interpolated)

    total_events_df = flx_interpolated * eff
    total_events_df['total'] = total_events_df['nu_e'] + total_events_df['nu_mu'] + total_events_df['nu_tau']

    if save_to_csv==True:
        total_events_df.to_csv('total_events.csv')

    return total_events_df



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
    
    # Bin width (assumed uniform; adjust if necessary)
    bin_width = energies[1] - energies[0]  # Assuming uniform binning
    
    # Loop over each energy bin
    for i, E_T in enumerate(energies):
        # Gaussian width depends on resolution and energy
        sigma = resolution * E_T  
        
        # Create Gaussian over all energy bins
        gaussian = np.exp(-0.5 * ((energies - E_T) / sigma) ** 2)
        gaussian /= gaussian.sum()  # Normalize Gaussian for proper redistribution
        
        # Redistribute current bin's events according to the Gaussian
        smeared_events += events[i] * gaussian
    
    return smeared_events



def rebinning(total_events, emin, emax, nbins):
    # Same procedure as used by HESE
    width = (np.log10(emax) - np.log10(emin)) / nbins
    e_edges, _, _ = binning.get_bins(emin, emax, ewidth=width, eedge=emin)
    bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))
    #log_bins = np.logspace(np.log10(1e5), np.log10(1e8), nbins+1)  # log-spaced bins

    # Group data into logarithmic bins
    total_events_binned = total_events.groupby(pd.cut(total_events.index, bin_centers)).sum()

    # Compute the midpoint (center) of each logarithmic interval
    # Geometric mean for log step midpoints
    total_events_binned['interval_center'] = [
        (interval.left * interval.right) ** 0.5 for interval in total_events_binned.index
    ]

    return total_events_binned



def fig10():

    eff, eff_err = plot_effective_areas()

    Edep1 = np.logspace(4, 5, num=20)
    Edep2 = np.logspace(5, 8, num=3*20+1)

    # Compute limited/extrapolated effective area and energy bins 
    eff_new1, energy_bins_new1 = get_effective_area_range(eff, Edep=Edep1, gen2=False)
    eff_new1 = np.asarray(eff_new1)

    eff_new2, energy_bins_new2 = get_effective_area_range(eff, Edep=Edep2, gen2=True)
    eff_new2 = np.asarray(eff_new2)

    # Set up pandas dataframes
    eff1_df = pd.DataFrame(eff_new1.T, index=energy_bins_new1, columns=['nu_e', 'nu_mu', 'nu_tau'])
    eff1_df.to_csv('Aeff_mockdata1.csv')

    eff2_df = pd.DataFrame(eff_new2.T, index=energy_bins_new2, columns=['nu_e', 'nu_mu', 'nu_tau'])
    eff2_df.to_csv('Aeff_mockdata2.csv')

    # Rebinn the MC mockdata
    mc1 = pd.read_csv('mc_Gen1.csv')
    mc2 = pd.read_csv('mc_Gen2.csv')



