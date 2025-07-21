import numpy as np
from Astrid.data_processing import get_weights
from Astrid.effective_area import apply_energy_smearing, bin_centers_to_edges, bin_edges_to_centers
from Astrid.config import PARAMS
import pandas as pd

def get_mc_histogram(Edep, livetime, gen2=False, save=False):
    """
    Get the histogram of MC events.
    Save the histogram of MC events to CSV file if specified.
    
    Parameters:
    -----------
    gen2 : bool
        If True, use Gen2 parameters, if False use Gen1 parameters
    """

    Edep_placeholder = np.logspace(np.log10(Edep[0]), np.log10(Edep[-1]), num=600)
    gen = 'Gen1' if not gen2 else 'Gen2'
    output_file = gen + '_smeared_si' + str(round(PARAMS[3], 2)) + '_norm' + str(round(PARAMS[4], 2)) + '.csv'

    # Get MC weights
    mc, weights = get_weights(Edep, livetime=livetime, gen2=gen2)
    
    # Create initial histogram
    weights_hist, _ = np.histogram(mc['recoDepositedEnergy'], bins=Edep_placeholder, weights=weights[0])
    
    # Apply energy smearing
    energies = bin_edges_to_centers(Edep_placeholder)
    mc_with_res = apply_energy_smearing(energies=energies,
                                      events=weights_hist,
                                      resolution=0.1)

    # Create final histogram
    mc_hist, edges = np.histogram(energies, bins=Edep, weights=weights_hist)
    mc_hist_smeared, edges = np.histogram(energies, bins=Edep, weights=mc_with_res)
    
    # Save to CSV
    mc_df = pd.DataFrame({'energy': edges[1:], 'events': mc_hist, 'with_resolution': mc_hist_smeared})
    if save:
        mc_df.to_csv(output_file, index=False)
    return mc_df
