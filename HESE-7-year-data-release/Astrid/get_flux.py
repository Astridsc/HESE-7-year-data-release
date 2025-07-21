from weighter import Weighter
from Astrid.effective_area import get_effective_area_dataframe
import numpy as np
from Astrid.data_processing import get_weights


# Parameter names
PARAMETER_NAMES = [
    "cr_delta_gamma", "nunubar_ratio", "anisotropy_scale", "astro_gamma",
    "astro_norm", "conv_norm", "epsilon_dom", "epsilon_head_on",
    "muon_norm", "kpi_ratio", "prompt_norm",
]

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
        0.00000001,   # Actual value is 0.0 but to avoid complications in get_weights
    ]
)

params = dict(zip(PARAMETER_NAMES, PARAMS))

# Define your energy range
energy_bins = np.logspace(4, 7, 3 * 20 + 1)  # Example: 10 TeV to 10 PeV
eff_df = get_effective_area_dataframe(energy_bins)

# Get MC events and weights
mc, weights = get_weights(energy_bins, livetime=227708167.68)  # HESE livetime


weight_maker = Weighter(mc)
astro_flux = weight_maker.flux_spl(mc, astro_norm=params[''], astro_gamma=2.5)
conv_flux = weight_maker.flux_conv(mc, conv_norm=1.0, kpi_ratio=1.0, cr_delta_gamma=0.0)
prompt_flux = weight_maker.flux_prompt(mc, prompt_norm=1.0, cr_delta_gamma=0.0)

# Convert to flux
flux = weights[0] / (eff_df * 227708167.68)  # Divide by effective area and livetime
