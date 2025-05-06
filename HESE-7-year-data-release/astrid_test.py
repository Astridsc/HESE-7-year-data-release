import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style

matplotlib.style.use("./resources/mpl/paper.mplstyle")
from matplotlib.font_manager import FontProperties
import scipy.stats

import data_loader
import weighter
import binning
import fc

import pandas as pd
import json

livetime = 227708167.68
#livetime = 315360000
emin_ = 60.0e3
emax_ = 1e7

def load_true_events(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.
mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
mc = data_loader.load_mc(mc_filenames, emin=emin_, emax=emax_)
data = data_loader.load_data("./resources/data/HESE_data.json", emin=emin_, emax=emax_)

e_edges, _, _ = binning.get_bins(emin=emin_, emax=emax_, ewidth=0.1, eedge=emin_)
bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

weight_maker = weighter.Weighter(mc)

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
weights = [weights[x] for x in range(len(weights))]



fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 9),
                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
#fit, ax = plt.subplots(2)
ax1.loglog()
plt.xlim(emin_, emax_)
#ax1.set_ylim(1.0e-1, 1.0e2)
ax1.set_ylabel("Events over 10 years")



ax1.hist(
    len(weights) * [mc["recoDepositedEnergy"]],
    weights=weights,
    bins=e_edges,
    histtype="bar",
    stacked=True,
    label='No ' + r'$\nu SI$',
    color=colors,
)
 

nuSI_df = pd.read_csv('Total_events.csv', index_col=0)


area_conversion = 1e4
norm = 1e-12

N_nuSI = nuSI_df['with_resolution'] * livetime *area_conversion * norm
energies = nuSI_df['interval_center']
print(N_nuSI)

ax1.step(energies, N_nuSI, label='With resolution', color='r')

"""ax1.hist(
    energies,
    weights=N_nuSI,
    bins=e_edges,
    log=True,
    histtype="step",
    label=r'$\nu SI$',
    color='r',
)"""


font = FontProperties()
font.set_size("medium")
font.set_family("sans-serif")
font.set_weight("bold")

# Simply reverses the order of the legend labels.
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1])
#ax1.tight_layout()


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
