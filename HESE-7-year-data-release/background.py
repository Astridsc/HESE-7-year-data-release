import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.style

import pandas as pd
#import binning
import json

import data_loader
import weighter
import binning
import fc

from Astrid.config import MC_FILENAMES, PARAMETER_NAMES, PARAMS
from Astrid.data_processing import load_true_events, get_particle_masks, get_weights
from Astrid.effective_area import bin_edges_to_centers, apply_energy_smearing





# Load data/MC. By default load_mc loads events at energies >60 TeV, but we want to plot all events.
#mc = data_loader.load_mc(MC_FILENAMES, emin=10.0e3)
data = data_loader.load_data("./resources/data/HESE_data.json", emin=10.0e3)


e_edges, _, _ = binning.get_bins(emin=10.0e3)
print(len(e_edges))
bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))
livetime = 12 * 365 * 24 * 3600  # 1 year in seconds

def get_background(e_edges, livetime):
    mc, weights = get_weights(Edep=e_edges, livetime=livetime, gen2=False, background=True)

    muon_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=e_edges, weights=weights[0])
    conv_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=e_edges, weights=weights[1])
    prompt_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=e_edges, weights=weights[2])
    astro_events,_ = np.histogram(mc['recoDepositedEnergy'], bins=e_edges, weights=weights[3])

    background_df = pd.DataFrame({'energy_edges[1:]': e_edges[1:], 'muon': muon_events, 'conv': conv_events, 'prompt': prompt_events, 'astro': astro_events})
    background_df.set_index('energy_edges[1:]', inplace=True)
    print(background_df)
    background_df.to_csv('background_df.csv')

    background = muon_events + conv_events + prompt_events
    print(f"Background events: {np.sum(background)}")
    hese12 = pd.read_csv('Astrid/HESE12_events.csv', index_col=0)
    #print(min(hese12['energy']), max(hese12['energy']))
    hese12_events, _ = np.histogram(hese12['energy'], bins=e_edges)
    print('hese12_events: ', hese12_events)

    astro_events12 = hese12_events - background
    print('astro_events12: ', astro_events12)
    astro_events12_smeared = apply_energy_smearing(energies=bin_centers, events=astro_events12, resolution=0.1)
    print('astro_events12_smeared: ', astro_events12_smeared)
    plt.stairs(
        astro_events12_smeared,
        e_edges,
        fill=False,
        label="HESE12(Data) - Background(MC), no smearing",
        color='lightgray')

    plt.stairs(
        astro_events12,
        e_edges,
        fill=False,
        label="HESE12(data) - Background(MC), smeared",
        color='r')

    plt.stairs(
        astro_events,
        e_edges,
        fill=False,
        label="7.5 years scaled to 12 (MC)",
        color='b')

    plt.stairs(
        background,
        e_edges,
        fill=False,
        label="Background scaled to 12 (MC muon + conv + prompt)",
        color='gray',
        linestyle='--'  
    )

    plt.xscale('log')
    plt.legend()
    plt.show()
    
    
    
def plot_background(e_edges):
    print(e_edges)
    background_df = pd.read_csv('background_df.csv', index_col=0)
    plt.stairs(
        (12/7.5)*background_df['astro'],
        e_edges,
        fill=False,
        label="Astro.",  
    )
    plt.stairs(
        (12/7.5)*background_df['conv'],
        e_edges,
        fill=False,
        label="Atmo. Conv.",
    )
    plt.stairs(
        (12/7.5)*background_df['prompt'],
        e_edges,
        fill=False,
        label="Atmo. Prompt.",
    )
    plt.stairs(
        (12/7.5)*background_df['muon'],
        e_edges,
        fill=False,
        label="Atmo. Muons.",
    )
    plt.xscale('log')
    plt.legend()
    plt.show()
    
plot_background(e_edges)