import sys
import os
import os.path
base_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(0, base_path + "/resources/external/")

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import pandas as pd
import numpy as np
import json
#import binning

import effective_area

df = pd.read_csv('HESE12_events.csv', index_col=0)



energies = df['energy']


bins = np.logspace(4, 7, 15)
counts, bin_edges = np.histogram(energies, bins=bins)

# Bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Poisson error bars (sqrt(N))
errors = np.sqrt(counts)

plt.errorbar(
    bin_centers, counts, yerr=errors, fmt='x', color='blue', label='Events',
    capsize=3, markersize=8, linestyle='none'
)
plt.xscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number of events')
plt.title('HESE12 Event Distribution (Crosses)')
plt.legend()
plt.show()






















"""energies = events_df['energy']
morph = events_df['event_morphology']

track_energies = energies[morph == 'Track']
print(f"Number of track events: {len(track_energies)}")
shower_energies = energies[morph == 'Shower']

shower_events, bin2, patches2 = plt.hist(shower_energies, bins=np.logspace(4, 7, 25), color='orange', alpha=0.5, label='Shower events')
track_events, bins1, patches1 = plt.hist(track_energies, bins=np.logspace(4, 7, 25), color='blue', alpha=0.5, label='Track events')
plt.xlabel('Energy [GeV]')
plt.ylabel('Number of events')
plt.xscale('log')
plt.title('HESE12 Event Distribution')
plt.legend()
plt.show()"""