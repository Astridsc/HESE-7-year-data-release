# Livetime constant
LIVETIME0 = 227708167.68  # The exact number of seconds for the 7.5 year run
LIVETIME1 = 7.5 * 365 * 24 * 3600  # 7.5 years in seconds
LIVETIME2 = 10 * 365 * 24 * 3600  # 10 years in seconds
LIVETIME3 = 15 * 365 * 24 * 3600  # 15 years in seconds

# File paths
MC_FILENAMES = [
    "/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/resources/data/HESE_mc_observable.json",
    "/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/resources/data/HESE_mc_flux.json", 
    "/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/resources/data/HESE_mc_truth.json",
]


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
        2.87375956,
        6.36488608,
        1.00621679,
        0.95192328,
        -0.0548763,
        1.18706341,
        1.00013744,
        0.00000000,   # Actual value is 0.0 but to avoid complications in get_weights
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