import sys
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.interpolate import interp1d
from scipy.special import factorial

import argparse
import time

#from Astrid.effective_area import apply_energy_smearing
#import weighter
#import weighter_original
import binning
#import data_loader
#import autodiff
#import likelihood
#import det_sys_weights
import sys
import os
import os.path
# Add nuSIprop to path (../../nuSIprop from this file's location)
base_path = os.path.dirname(os.path.abspath(__file__))
nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop-main-new'))
if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
import nuSIprop



def get_MESE_MC_events(model):
    # Load the MC events for astrophysical flux
    # Either for BPL (Fig. 6) or SPL (Fig. 8)
    if model == 'BPL':
        cascade_events = pd.read_csv('MESE_figures/cascades/MC_astroBPL_cascades_MESE.csv', header=None)
        track_events = pd.read_csv('MESE_figures/tracks/MC_astroBPL_tracks_MESE.csv', header=None)  
    elif model == 'SPL':
        cascade_events = pd.read_csv('MESE_figures/cascades/MC_astroSPL_cascades_MESE.csv', header=None)
        track_events = pd.read_csv('MESE_figures/tracks/MC_astroSPL_tracks_MESE.csv', header=None) 
    else:
        raise ValueError(f"Invalid model: {model}")
    return cascade_events, track_events
        
        
def get_segmented_flux():
    
    energy_edges_MESE = [1.0, 2.15, 4.64, 10.0, 21.5, 46.4, 100.0, 215.4, 464.2, 1000.0, 2154.4, 4641.6, 10000.0, 100000.0]
    energy_centers_MESE = [1.47, 3.16, 6.81, 14.7, 31.6, 68.1, 146.8, 316.2, 681.3, 1467.8, 3162.3, 6812.9, 31622.8]
    # MESE 
    norm_segmented = [0.0, 5.4, 3.9, 4.41, 5.51, 3.34, 1.55, 0.31, 0.59, 0.327, 0.0, 0.25, 0.0]
    sigma_upper = [5.2, 5.8, 2.1, 0.93, 0.72, 0.63, 0.50, 0.33, 0.42, 0.32, 0.19, 0.24, 0.76]
    sigma_lower = [0.0, 5.4, 2.0, 0.90, 0.66, 0.52, 0.35, 0.31, 0.23, 0.187, 0.0, 0.16, 0.0]
    
    energy_edges_MESE, energy_centers_MESE, norm_segmented, sigma_upper, sigma_lower = np.asarray(energy_edges_MESE), np.asarray(energy_centers_MESE), np.asarray(norm_segmented), np.asarray(sigma_upper), np.asarray(sigma_lower)
    energy_edges_MESE *= 1e3 # GeV
    energy_centers_MESE *= 1e3 # GeV

    E0 = 1e5    # if E in GeV; 100 TeV = 1e5 GeV
    norm_segmented *= 1e-18 * E0**2   # GeV^-1 * cm^-2 * s^-1 * sr^-1
    sigma_upper *= 1e-18 * E0**2   # GeV^-1 * cm^-2 * s^-1 * sr^-1
    sigma_lower *= 1e-18 * E0**2   # GeV^-1 * cm^-2 * s^-1 * sr^-1
    return energy_edges_MESE, energy_centers_MESE, norm_segmented, sigma_upper, sigma_lower


def get_reconstructed_flux(energy_edges_MESE, energy_centers_MESE, norm_segmented, morphology):
    # Convert the flux from segmented (in terms of E_nu) to reconstructed (in terms of E_rec)
    # Re-bin the flux, and apply appropriate smearing
    
    if morphology == 'c':
        resolution = 0.11
        E_rec_edges = np.logspace(3, 4, 22+1)
    elif morphology == 't':
        resolution = 0.30
        E_rec_edges = np.logspace(3, 7, 13+1)
    
    flux_reconstructed = np.zeros(len(E_rec_edges)-1)
    
    #integrand_fine_binning = np.zeros(len(energy_centers_MESE) * len(E_rec_edges) - 1)
    energies_fine_binning = np.zeros(len(energy_centers_MESE) * len(E_rec_edges) - 1)    # Något oklart exakt hur lång denna bör va, men det kanske inte spelar någon roll
    norms = np.zeros(len(energy_centers_MESE) * len(E_rec_edges) - 1)
    #dummy_counter = 0
    
    for i, E in enumerate(energies_fine_binning):
        
        for k, MESE_edge in enumerate(energy_edges_MESE[:-1]):
            if E >= MESE_edge and E < energy_edges_MESE[k+1]:
                norms[i] = norm_segmented[k]
                break
    print('norms: ', norms)
    integrand = apply_energy_smearing(energies=energies_fine_binning, events=norms*energies_fine_binning**(-2), resolution=resolution)
    reconstructed_flux = np.trapz(integrand, energies_fine_binning)
    binned_reconstructed_flux, _ = np.histogram(energies_fine_binning, weights=integrand, bins=E_rec_edges)
    return binned_reconstructed_flux
    



def approximate_effective_area(MESE_MC_events, reconstructed_flux):
    # This is an approximation of the effective area, that should account for the time factor as well (+ any other conversion factor like 4*pi)
    return MESE_MC_events / reconstructed_flux
    

    
    

        
        
            
                




        