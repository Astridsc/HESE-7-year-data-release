%matplotlib inline
import sys
import os
import os.path

#base_path = os.path.dirname(os.path.abspath(__file__))
base_path = '/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release'
sys.path.insert(0, base_path + "/resources/external/")

nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop-main-new'))
if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
import nuSIprop

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import matplotlib
#import matplotlib.style
from Astrid.effective_area import bin_edges_to_centers, apply_energy_smearing, 
from Astrid.config import MC_FILENAMES, PARAMS, PARAMETER_NAMES

params_dict = dict(zip(PARAMETER_NAMES, PARAMS))
#matplotlib.style.use(os.path.join(base_path, "resources/mpl/paper.mplstyle"))
#matplotlib.use("Agg") 
#from matplotlib.font_manager import FontProperties
import scipy.stats
import weighter_original
import weighter_original_original
import data_loader
import weighter
import binning
import fc


cascade_energy_bins = np.logspace(3, 7, 22+1)
track_energy_bins = np.logspace(3, 7, 13+1)
casc_energy_centers = bin_edges_to_centers(cascade_energy_bins)
track_energy_centers = bin_edges_to_centers(track_energy_bins)

e_edges_casc, _, _ = binning.get_bins(emin=1e3, emax=1e7, ewidth=0.18181818181818181818181818181818)
bin_centers_casc = 10.0 ** (0.5 * (np.log10(e_edges_casc[:-1]) + np.log10(e_edges_casc[1:])))
e_edges_track, _, _ = binning.get_bins(emin=1e3, emax=1e7, ewidth=0.30769230769230769230769230769231)
bin_centers_track = 10.0 ** (0.5 * (np.log10(e_edges_track[:-1]) + np.log10(e_edges_track[1:])))

livetime_75 = 227708167.68  


def center(x):
    x = np.asarray(x)
    return (x[1:] + x[:-1]) / 2.0

energy_bins = np.logspace(2, 7, 5 * 20 + 1)  # 1e2 to 1e7 with 20 bins per decade
energy_bin_widths = np.diff(energy_bins)
energy_centers = center(energy_bins)

def BPL_flux(energies=energy_centers, norm=2.28, si1=1.72, si2=2.839, E_break=10**4.524):
    # BPL model flux
    # Normalized at E = 100TeV = 1e5 GeV
    # Best fit MESE: norm = 2.28, si1 = 1.72, si2 = 2.839, E_break = 10**(4.524) GeV
    norm = norm*1e-18
    E0 = 1e5
    if E_break > 1e5:
        norm_break  = norm * (E_break / E0)**(-si1)
    else:
        norm_break  = norm * (E_break / E0)**(-si2)
    
    # If used for integration within the bin, energies will be a single value.
    if isinstance(energies, float):
        if energies < E_break:
            flux = norm_break * (energies / E0)**(-si1)
        else:
            flux = norm_break * (energies / E0)**(-si2)
    else:
        flux = np.zeros(len(energies))
        for i, E in enumerate(energies):
            if E < E_break:
                flux[i] = norm_break * (E / E_break)**(-si1)
            else:
                flux[i] = norm_break * (E / E_break)**(-si2)
    return flux 

    
def LP_flux(energies=energy_centers, norm=2.58, alpha=2.669, beta=0.359):
    # Log Parabola model flux
    # Normalized at E = 100TeV
    # Best fit MESE: norm = 2.42, alpha = 2.05, beta = 2.54
    E0 = 1e5
    norm = norm*1e-18
    return  norm * (energies / E0)**(-alpha - beta * np.log10(energies / E0)) 

def SPL_flux(energies=energy_centers, norm=2.28, si=2.548):
    # Single Power Law model flux
    # Normalized at E = 100TeV
    # Best fit MESE: norm = 2.28, si = 1.72
    E0 = 1e5
    return norm * (energies / E0)**(-si)


def get_HESE_MC():
    mc = data_loader.load_mc(MC_FILENAMES, emin=1e3, emax=1e7)
    
    mc_cascades = mc[mc['recoMorphology'] == 0]
    mc_tracks = mc[mc['recoMorphology'] == 1]
    
    weight_maker_cascades = weighter_original.Weighter(mc_cascades, model='spl')
    weight_maker_tracks = weighter_original.Weighter(mc_tracks, model='spl')
    
    weights_cascades = weight_maker_cascades.get_weights(livetime=livetime_75, parameter_names=PARAMETER_NAMES, params=PARAMS)
    weights_tracks = weight_maker_tracks.get_weights(livetime=livetime_75, parameter_names=PARAMETER_NAMES, params=PARAMS)
    
    binned_cascade_weights, _ = np.histogram(mc_cascades, cascade_energy_bins, weights=weights_cascades[0])
    binned_track_weights, _ = np.histogram(mc_tracks, track_energy_bins, weights=weights_tracks[0])
    
    return binned_cascade_weights, binned_track_weights


def get_MESE_MC():
    mc_casc_tot_SPL_MESE = np.asarray(pd.read_csv('sandhyas_bullshit/cascades/MC_tot_cascades_SPL_MESE.csv', usecols=[1], header=None))
    mc_track_tot_SPL_MESE = np.asarray(pd.read_csv('sandhyas_bullshit/tracks/MC_tot_tracks_SPL_MESE.csv', usecols=[1], header=None))
    
    return mc_casc_tot_SPL_MESE, mc_track_tot_SPL_MESE


def scale_events(HESE_mc, MESE_mc):
    scale_factor = np.zeros(HESE_mc.shape())
    for i, event in enumerate(HESE_mc):
        
        scale_factor[i] = MESE_mc[i] / event
        
    return scale_factor



def initialize_nuSIprop(Mphi_val, g_val, si_val, mntot_val):
    """
    Initialize nuSIprop object with given parameters.
    
    Parameters:
    -----------
    Mphi_val : float
        Mphi parameter in GeV
    g_val : float
        g parameter
    si_val : float
        Spectral index (astro_gamma)
    mntot_val : float
        mntot parameter
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(nuSIprop_path)
        evolver = nuSIprop.pyprop(
            mphi=Mphi_val*1e6,  # Convert GeV to eV
            g=g_val,
            si=si_val,
            norm=1e-18,
            mntot=mntot_val,
            majorana=True,
            non_resonant=True,
            normal_ordering=True,
            N_bins_E=300,
            lEmin=12-0.01,
            lEmax=17.01,
            zmax=4,
            flav=2,
            phiphi=True
            )
        evolver.evolve()
    finally:
        os.chdir(original_cwd)
    return evolver







casc_HESE, track_HESE = get_HESE_MC()
casc_MESE, track_MESE = get_MESE_MC()

scale_factor_casc = scale_factor(casc_HESE, casc_MESE)
scale_factor_track = scale_factor(track_HESE, track_MESE)

Aeff_casc = pd.read_csv('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/effective_areas/All flavors cascade_effective_area.csv')['effective_area_m2'].iloc[0:].reset_index(drop=True)
Aeff_track = pd.read_csv('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/effective_areas/All flavors track_effective_area.csv')['effective_area_m2'].iloc[0:].reset_index(drop=True)
Aeff_dcasc = pd.read_csv('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/effective_areas/All flavors double_cascade_effective_area.csv')['effective_area_m2'].iloc[0:].reset_index(drop=True)
Aeff_energies = pd.read_csv('/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/effective_areas/All flavors cascade_effective_area.csv')['energy_bins'].iloc[0:].reset_index(drop=True)

spl_flux = SPL_flux(energies=Aeff_energies)


casc_smeared = apply_energy_smearing(energies=np.asarray(Aeff_energies), events=spl_flux * np.asarray(Aeff_casc), resolution=0.11)
track_smeared = apply_energy_smearing(energies=np.asarray(Aeff_energies), events=spl_flux * np.asarray(Aeff_tracks), resolution=0.30)

