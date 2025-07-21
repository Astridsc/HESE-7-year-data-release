import sys
import os
import os.path

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path + "/resources/external/")

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.style
#matplotlib.style.use("./resources/mpl/paper.mplstyle")
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors

import scipy.stats
from scipy.interpolate import interp1d
import numpy as np

import data_loader
import weighter
import binning
#import fc
import functools

import json
import pandas as pd
from Astrid.config import MC_FILENAMES, PARAMETER_NAMES, PARAMS
from Astrid.config import LIVETIME1, LIVETIME2, LIVETIME3




def load_true_events(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data



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


def get_data(Edep):
    data = data_loader.load_data("./resources/data/HESE_data.json", emin=Edep[0], emax=Edep[-1])
    return data


def get_weights(Edep, livetime, gen2=False, background=False):
    """
    Calculate weights for Monte Carlo events based on energy deposition and livetime.

    Parameters:
        Edep (array-like): Energy deposition range (e.g., np.logspace(emin, emax, n)).
        livetime (float): Detector livetime in seconds.
        gen2 (bool): Whether to apply Gen2 scaling (factor of 10). Default is False.

    Returns:
        tuple: (mc, weights)
    """
    # Load the data
    mc = data_loader.load_mc(MC_FILENAMES, emin=Edep[0], emax=Edep[-1])    # Usually on the order of 1e5 events?...
    data = data_loader.load_data("/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/resources/data/HESE_data.json", emin=Edep[0], emax=Edep[-1])

    e_edges, _, _ = binning.get_bins(emin=Edep[0], emax=Edep[-1])
    #bin_centers = 10.0 ** (0.5 * (np.log10(e_edges[:-1]) + np.log10(e_edges[1:])))

    #n_events, _ = np.histogram(data["recoDepositedEnergy"], bins=e_edges)

    weight_maker = weighter.Weighter(mc)

    if background == False:
        component_order = [
            ("astro_norm", "Astro."),
        ]
    else:
        component_order = [
            ("muon_norm", "Atmo. Muons"),
            ("conv_norm", "Atmo. Conv."),
            ("prompt_norm", "Atmo. Prompt"),
            ("astro_norm", "Astro.")
        ]

    params_dict = dict(zip(PARAMETER_NAMES, PARAMS))
    params_zeroed = params_dict.copy()

    for (zeroed_norm, _) in component_order:
        params_zeroed[zeroed_norm] = 0.0

    # We want to separate the histogram by components, so we separately get the weights
    # where the all normalization parameters but one are set to zero
    weights = []
    #colors = []
    labels = []
    #cm = plt.get_cmap("magma")
    #color_scale = [cm(x) for x in [0.75]]

    for i, (zeroed_norm, zeroed_label) in enumerate(component_order):
        if params_dict[zeroed_norm] == 0.0:
            continue
        p_copy = params_zeroed.copy()
        p_copy[zeroed_norm] = params_dict[zeroed_norm]
        weights.append(
            weight_maker.get_weights(livetime, p_copy.keys(), p_copy.values())[0]
        )
        #colors.append(color_scale[i])
        labels.append(zeroed_label)

    #Factor 10 to account for increase in effective area, which is approximately 10, according to Fig.25, Ref. 49
    if gen2==True:
        A_factor = 10
    else:
        A_factor = 1
    weights = [A_factor * weights[x] for x in range(len(weights))]
    
    return mc, weights


def get_muon_weights(Edep, livetime, gen2=False):
    """
    Calculate weights for Monte Carlo events based on energy deposition and livetime.

    Parameters:
        Edep (array-like): Energy deposition range (e.g., np.logspace(emin, emax, n)).
        livetime (float): Detector livetime in seconds.
        gen2 (bool): Whether to apply Gen2 scaling (factor of 10). Default is False.

    Returns:
        tuple: (mc, weights)
    """
    # Load the data
    mc = data_loader.load_mc(MC_FILENAMES, emin=Edep[0], emax=Edep[-1])
    data = data_loader.load_data("/home/astridaurora/HESE-7-year-data-release/HESE-7-year-data-release/resources/data/HESE_data.json", emin=Edep[0], emax=Edep[-1])

    weight_maker = weighter.Weighter(mc)
    
    n_params = len(PARAMS)

    p = dict()

    # Initialize parameter vector with gradient
    for i, (name, param) in enumerate(zip(PARAMETER_NAMES, PARAMS)):
        p_grad = np.zeros(shape=n_params).astype(float)
        p_grad[i] = 1.0
        p[name] = [param, p_grad]
    
    muon_norm = p["muon_norm"]
    print(muon_norm)

    #params_dict = dict(zip(PARAMETER_NAMES, PARAMS))
    
    # Get muon weights
    weights = weight_maker.weight_muon(mc, muon_norm=muon_norm)
    
    # Apply Gen2 scaling if requested
    if gen2:
        A_factor = 10
    else:
        A_factor = 1
    
    # Apply livetime and scaling factor
    weights = A_factor * livetime * weights
    
    # Return as a list to maintain compatibility with existing code
    return mc, weights