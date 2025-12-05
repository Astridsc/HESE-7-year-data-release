import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style

matplotlib.style.use(os.path.join(base_path, "resources/mpl/paper.mplstyle"))
matplotlib.use("TkAgg") 
from matplotlib.font_manager import FontProperties
import scipy.stats

import data_loader
import weighter
import binning
import fc


def _apply_energy_smearing_serial(energies, events, resolution):
    """Serial version of energy smearing"""
    smeared_events = np.zeros_like(events)  
    if energies.any() == 0:
        print(f"Warning: Energies are zero at index", np.where(energies == 0))
    else:
        print(f"Energies are not zero")
        
    
    for i, E_true in enumerate(energies):
        # Calculate sigma in linear space (resolution is fractional)
        sigma = resolution * E_true
        
        # Create Gaussian in linear space
        gaussian = np.exp(-0.5 * ((energies - E_true) / sigma) ** 2)
        
        # Normalize the Gaussian
        gaussian_sum = np.sum(gaussian)
        if gaussian_sum > 0:  # Avoid division by zero
            gaussian /= gaussian_sum
        else:
            print(f"Warning: Gaussian sum is zero for energy {E_true}")
        
        # Redistribute events
        smeared_events += events[i] * gaussian
    
    # Verify event conservation
    total_events_before = np.sum(events)
    total_events_after = np.sum(smeared_events)
    if not np.isclose(total_events_before, total_events_after, rtol=1e-10):
        print(f"Warning: Event conservation violated! Before: {total_events_before}, After: {total_events_after}")
    
    return smeared_events

# Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.
mc_filenames = [
    os.path.join(base_path, "resources/data/HESE_mc_observable.json"),
    os.path.join(base_path, "resources/data/HESE_mc_flux.json"),
    os.path.join(base_path, "resources/data/HESE_mc_truth.json"),
]
mc = data_loader.load_mc(mc_filenames, emin=10.0e3)
import nuSIprop
nuSIprop = nuSIprop.pyprop(
    mphi=5*1e6, g=0.1, si=2.0, norm=1, mntot=0.1,
    majorana=True, non_resonant=True, normal_ordering=True,
    N_bins_E=300, lEmin=13, lEmax=16.01, zmax=5, flav=2, phiphi=False
)
weight_maker = weighter.Weighter(mc, model="nusiprop", nuSIprop=nuSIprop, simple=True)
print('Weight maker initialized')


data = data_loader.load_data(os.path.join(base_path, "resources/data/HESE_data.json"), emin=10.0e3)
data12 = data_loader.load_data(os.path.join(base_path, "resources/data/HESE12_data.json"), emin=10.0e3)

e_edges, _, _ = binning.get_bins(emin=10.0e3)
bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))
e_edges_medium_resolution, _, _ = binning.get_bins(emin=10.0e3, ewidth=0.00111)
bin_centers_medium_resolution = 10.0 ** (0.5 * (np.log10(e_edges_medium_resolution[:-1]) + np.log10(e_edges_medium_resolution[1:])))

n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

"""plt.hist(mc["recoDepositedEnergy"],
         bins=e_edges, 
         histtype="step", 
         stacked=False, 
         label="MC events", 
         color="teal",
         alpha=0.8)
plt.xscale('log')
plt.xlabel("Deposited Energy [GeV]")
plt.ylabel("Number of events")
plt.legend()
plt.grid(axis='both', which='both', linestyle='--', linewidth=0.5)
plt.show()
plt.close()"""



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

# 2.87375956 best fit spectral index
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

livetime = 227708167.68
params_dict = dict(zip(parameter_names, params))
params_zeroed = params_dict.copy()

component_order = [
    ("muon_norm", "Atmo. Muons"),
    ("conv_norm", "Atmo. Conv."),
    ("prompt_norm", "Atmo. Prompt"),
    ("astro_norm", "Astro."),
]

for (zeroed_norm, _) in component_order:
    params_zeroed[zeroed_norm] = 0.0

# We want to separate the histogram by components, so we separately get the weights
# where the all normalization parameters but one are set to zero
weights = []
colors = []
labels = []
cm = plt.get_cmap("inferno")
color_scale = [cm(x) for x in [0.2, 0.55, 0.75, 0.9]]
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

fit, ax = plt.subplots(figsize=(7, 5))
plt.loglog()
plt.xlim(10.0e3, 10.0e6)
plt.ylim(1.0e-1, 4.0e1)
plt.xlabel("Deposited Energy [GeV]")
plt.ylabel("Events per 2635 days")

xerr = [bin_centers - e_edges[:-1], e_edges[1:] - bin_centers]
# The error bars are obtained using fc.py, a function writen by Austin Schneider
# and found at https://github.com/austinschneider/feldman_cousins/blob/master/fc.py
one_sigma_proportion = scipy.special.erf(1.0 / np.sqrt(2.0))
yerr = np.array(
    [
        (lambda x: [k - x[0], x[1] - k])(
            fc.poisson_interval(k, alpha=one_sigma_proportion)
        )
        for k in n_events
    ]
).T

plt.errorbar(
    bin_centers,
    n_events,
    xerr=xerr,
    yerr=yerr,
    color="black",
    marker=None,
    label="Data",
    linestyle="None",
    capsize=3,
    elinewidth=1,
)

atmospheric_weights = weights[0] + weights[1] 
weights = [atmospheric_weights, weights[2]]
#colors = [colors[0], 'red']
plt.hist(
    len(weights) * [mc["recoDepositedEnergy"]],
    weights=weights,
    bins=e_edges,
    histtype="bar",
    stacked=True,
    label=["Atmo.", "nuSIprop."],
)

#plt.step(bin_centers, weights[-1], color='red', label="nuSIprop.")

# This applies the visual filter that covers events that are not used
# in the analysis
mask_color = "#7ab9f3"
ax.fill_between(
    [10e3, 60e3],
    [1e3, 1e3],
    [0, 0],
    edgecolor="none",
    linewidth=0.0,
    facecolor=mask_color,
    zorder=3,
    alpha=0.7,
)
plt.axvline(x=60e3, linestyle="dashed")

font = FontProperties()
font.set_size("medium")
font.set_family("sans-serif")
font.set_weight("bold")

# Simply reverses the order of the legend labels.
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

plt.tight_layout()
plt.savefig('test_diffuse_energy_projection.png')
plt.show()
plt.close()


plt.hist(mc["recoDepositedEnergy"], weights=weights[-1], 
         bins=e_edges, 
         histtype="bar", 
         stacked=True,
         label=labels[-1], 
         color='green')

"""
events_medium_resolution, _ = np.histogram(mc["recoDepositedEnergy"], bins=e_edges_medium_resolution, weights=weights[-1])
energies_medium_resolution, _ = np.histogram(mc["recoDepositedEnergy"], bins=e_edges_medium_resolution)
smeared_events = _apply_energy_smearing_serial(energies_medium_resolution, events_medium_resolution, resolution=0.1)

plt.hist(bin_centers_medium_resolution, weights=events_medium_resolution, 
         bins=e_edges, 
         histtype="step", 
         stacked=False,
         label="No smearing",
         color="lightgray",
         linestyle="--")

plt.hist(bin_centers_medium_resolution, weights=smeared_events, 
         bins=e_edges, 
         histtype="step", 
         stacked=False,
         label="Smeared",
         color="red")"""

plt.xlim(6*1e4, 1e7)
plt.ylim(0, 14.5)
plt.xscale('log')
plt.xlabel("Deposited Energy [GeV]")
plt.ylabel("Events per 2635 days")
plt.savefig('test_Fig6.png')
plt.show()
plt.close()



"""

atmospheric_weights = weights[0] + weights[1] + weights[2]
weights = [atmospheric_weights, weights[-1]]

plt.hist(mc["recoDepositedEnergy"], weights=weights, 
         bins=e_edges, 
         histtype="bar", 
         stacked=True,
         label=labels, 
         color=colors)
plt.xlim(6*1e4, 1e7)
plt.ylim(1e-1, 3*1e1)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Deposited Energy [GeV]")
plt.ylabel("Events per 2635 days")
plt.savefig('test_Fig6_astro.png')
plt.show()
plt.close()
"""