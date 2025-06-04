# Livetime constant
LIVETIME1 = 7.5 * 365 * 24 * 3600  # 7.5 years in seconds
LIVETIME2 = 10 * 365 * 24 * 3600  # 10 years in seconds
LIVETIME3 = 15 * 365 * 24 * 3600  # 15 years in seconds

# File paths
MC_FILENAMES = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]
FLUX_FILE_6 = "./resources/data/flux_Fig6.csv"
FLUX_FILE_8 = "./resources/data/flux_Fig8.csv"
MC_GEN1_FILE = "./resources/data/mc_Gen1.csv"
MC_GEN2_FILE = "./resources/data/mc_Gen2.csv"

# Parameter names
PARAMETER_NAMES = [
    "cr_delta_gamma", "nunubar_ratio", "anisotropy_scale", "astro_gamma",
    "astro_norm", "conv_norm", "epsilon_dom", "epsilon_head_on",
    "muon_norm", "kpi_ratio", "prompt_norm",
]

import numpy as np
# Parameters for the model
#2.87375956 best fit spectral index
# 6.36488608 best fit for astro norm
PARAMS = np.array(
    [
        -0.05309302,
        0.99815326,
        1.000683,
        2.9,
        6.36488608,
        1.00621679,
        0.95192328,
        -0.0548763,
        1.18706341,
        1.00013744,
        0.0,
    ]
)

# Plot settings
PLOT_SETTINGS = {
    "figsize": (7, 5),
    "xscale": "log",
    "yscale": "log",
    "xlabel": "Energy [GeV]",
    "ylabel": "Number of events",
}