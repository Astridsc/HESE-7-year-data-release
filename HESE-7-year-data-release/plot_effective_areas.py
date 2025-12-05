import sys
import os
import os.path
import matplotlib
import matplotlib.style

matplotlib.use("Agg")
matplotlib.style.use("./resources/mpl/paper.mplstyle")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import functools

import json
import pandas as pd

outdir = "./effective_areas/"

# Disable LaTeX text rendering
plt.rcParams['text.usetex'] = False


def center(x):
    x = np.asarray(x)
    return (x[1:] + x[:-1]) / 2.0


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


mc_filenames = [
    "./resources/data/HESE_mc_observable.json",
    "./resources/data/HESE_mc_flux.json",
    "./resources/data/HESE_mc_truth.json",
]


def plot_effective_areas(json_files=mc_filenames):
    try:
        os.makedirs(outdir)
    except:
        pass

    # Load the MC
    json_data = dict()
    for filename in json_files:
        json_data.update(json.load(open(filename, "r")))

    # Get the MC generation information
    weight_over_flux_over_livetime = np.array(json_data["weightOverFluxOverLivetime"])
    muon_weight_over_flux_over_livetime = np.array(json_data["muonWeightOverFluxOverLivetime"])

    # Choose the energy binning
    #energy_bins = np.logspace(2, 7, 5 * 20 + 1)  # 1e2 to 1e7 with 20 bins per decade
    energy_bins = np.logspace(3, 7, 22+1)
    energy_bin_widths = np.diff(energy_bins)

    # Get neutrino interaction information from the file
    primaryEnergy = np.array(json_data["primaryEnergy"])
    interactionType = np.array(json_data["interactionType"])
    primaryType = np.array(json_data["primaryType"])
    recoMorphology = np.array(json_data["recoMorphology"])

    # Get some masks that correspond to our chosen energy bins
    nu_energy_mapping = np.digitize(primaryEnergy, bins=energy_bins) - 1
    nu_energy_masks = [nu_energy_mapping == i for i in range(len(energy_bins) - 1)]

    # Get some masks that sort by interaction type
    interaction_types = [1, 2, 3]
    interaction_masks = [interactionType == i for i in interaction_types]
    CC_mask, NC_mask, GR_mask = interaction_masks
    
    morphology_types = [0, 1, 2]
    morphology_masks = [recoMorphology == i for i in morphology_types]
    cascade_mask, track_mask, double_cascade_mask = morphology_masks

    # Get some masks that sort by primary particle type
    # Remember these are the relevant entries in the dictionary:
    """
        'nue',   'nuebar',   '2nue',
        'numu',  'numubar',  '2numu',
        'nutau', 'nutaubar', '2nutau',
        'mu', 'nu', 'all',
    """
    particle_masks = get_particle_masks(primaryType)

    ## Now we have what we need to compute the effective area ##

    # Choose the color map
    cm = plt.get_cmap("plasma")

    # Choose some line styles
    line_styles = ["-", "--", ":", ":", "-", "--", ":", ":"]

    # 3 flavors in the MC
    n_flavors = 3

    # We are going to average our effective area over the whole sky
    total_angular_width = 4.0 * np.pi

    bin_widths = energy_bin_widths * total_angular_width

    # A meter is 100cm
    meter = 100

    # Function to compute effective area with custom energy binning and return dataframe
    def compute_effective_area_df(particle_mask, custom_energy_bins=None, factor=1.0):
        """
        Compute effective area for a given particle mask with custom energy binning.
        Returns a dataframe with energy bins, effective area, and errors.
        
        Args:
            particle_mask: Boolean mask for particles to include
            custom_energy_bins: Optional custom energy bins (default: uses global energy_bins)
            factor: Factor to multiply effective area by (default: 1.0)
        
        Returns:
            DataFrame with columns: energy_bins, effective_area_m2, effective_area_m2_error
        """
        # Use custom energy bins if provided, otherwise use global
        if custom_energy_bins is None:
            custom_energy_bins = energy_bins
        
        custom_energy_bin_widths = np.diff(custom_energy_bins)
        
        # We are going to average our effective area over the whole sky
        total_angular_width = 4.0 * np.pi
        custom_bin_widths = custom_energy_bin_widths * total_angular_width
        
        # Create energy masks for custom bins
        nu_energy_mapping = np.digitize(primaryEnergy, bins=custom_energy_bins) - 1
        custom_nu_energy_masks = [nu_energy_mapping == i for i in range(len(custom_energy_bins) - 1)]
        
        # Combine particle mask with energy masks
        masks = np.logical_and(particle_mask[None, :], custom_nu_energy_masks)
        
        # Effective area is the sum of weightOverFluxOverLivetime, divided by bin width
        effective_area_cm2 = np.array(
            [
                #np.sum(weight_over_flux_over_livetime[mask]) / bin_width
                np.sum(muon_weight_over_flux_over_livetime[mask]) / bin_width
                for mask, bin_width in zip(masks, custom_bin_widths)
            ]
        ) * factor

        # Compute the error on this quantity
        effective_area_cm2_error = np.array(
            [
                #np.sqrt(np.sum(weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                np.sqrt(np.sum(muon_weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                for mask, bin_width in zip(masks, custom_bin_widths)
            ]
        ) * factor

        # Convert to meters^2
        effective_area_m2 = effective_area_cm2 / (meter ** 2)
        effective_area_m2_error = effective_area_cm2_error / (meter ** 2)
        
        # Create dataframe
        df = pd.DataFrame({
            "energy_bins": center(np.log10(custom_energy_bins)),
            "effective_area_m2": effective_area_m2,
            "effective_area_m2_error": effective_area_m2_error
        })
        
        return df

    # How to compute and plot the effective area (in a histogram style with errors)
    def plot_line(ax, masks, color, line_style, label, factor=1.0, energy_bins_plot=None):
        # Each entry in masks corresponds to an energy bin
        # The mask should define the events that contribute to the effective area calcualtion in that bin
        if energy_bins_plot is None:
            energy_bins_plot = energy_bins
            bin_widths_plot = bin_widths
        else:
            bin_widths_plot = np.diff(energy_bins_plot) * total_angular_width

        # Effective area is the sum of weightOverFluxOverLivetime, divided by bin width
        effective_area_cm2 = np.array(
            [
                #np.sum(weight_over_flux_over_livetime[mask]) / bin_width
                np.sum(muon_weight_over_flux_over_livetime[mask]) / bin_width
                for mask, bin_width in zip(masks, bin_widths_plot)
            ]
        ) * factor
        # An additional factor may be needed if we are computing an average
        # effective area for multiple particle types

        # Compute the error on this quantity
        effective_area_cm2_error = np.array(
            [
                #np.sqrt(np.sum(weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                np.sqrt(np.sum(muon_weight_over_flux_over_livetime[mask] ** 2)) / bin_width
                for mask, bin_width in zip(masks, bin_widths_plot)
            ]
        ) * factor

        # Convert to meters^2
        effective_area_m2 = effective_area_cm2 / (meter ** 2)
        effective_area_m2_error = effective_area_cm2_error / (meter ** 2)
        print(effective_area_m2)

        # Plot things only if they will appear on the plot
        if np.any(effective_area_m2 > 1e-4):
            """# Make plot of effective area
            ax.step(
                energy_bins_plot[1:],
                effective_area_m2,
                color=color,
                linestyle=line_style,
                lw=2,
                label=label,
            )
            # Add the errorbars to the plot
            ax.errorbar(
                10 ** center(np.log10(energy_bins_plot)),
                effective_area_m2,
                yerr=effective_area_m2_error,
                color=color,
                linestyle="none",
            )"""
            
            # Create and return dataframe
            df = pd.DataFrame({
                "energy_bins": 10**center(np.log10(energy_bins_plot)),
                "effective_area_m2": effective_area_m2,
                "effective_area_m2_error": effective_area_m2_error
            })
            df.to_csv(outdir + "/" + label + "_effective_area.csv", index=False)
            print(df)
            print(label)
            return df
        return None

    # How to format the axis
    def format_axis(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim((1e4, 1e7))
        ax.set_ylim((1e-4, 2e3))
        ax.set_xlabel("Neutrino Energy [GeV]")
        ax.set_ylabel("Effective Area [m²]")

        # Override the yaxis tick settings
        major = 10.0 ** np.arange(-3, 5)
        minor = np.arange(2, 10) / 10.0
        locmaj = matplotlib.ticker.FixedLocator(10.0 ** np.arange(-2, 4))
        locmin = matplotlib.ticker.FixedLocator(
            np.tile(minor, len(major)) * np.repeat(major, len(minor))
        )
        locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1,), numticks=12)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=minor, numticks=12)
        ax.yaxis.set_major_locator(locmaj)
        ax.yaxis.set_minor_locator(locmin)

        ax.legend(frameon=True, loc="upper left")

    # How to save the figure with nice spacing
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(outdir + "/" + name + ".pdf")
        fig.savefig(outdir + "/" + name + ".png", dpi=400)
        fig.clf()
    
    # Make an effective area vs. energy plot split by morphology only (all flavors combined)
    # Define custom energy binning for each morphology (can be modified as needed)
    # Example: You can set different binning for each morphology:
    # morphology_energy_bins = {
    #     "cascade": np.logspace(3, 7, 30+1),  # More bins for cascade
    #     "track": np.logspace(3, 7, 22+1),    # Default bins for track
    #     "double_cascade": np.logspace(4, 7, 15+1)  # Different range for double_cascade
    # }
    morphology_energy_bins = {
        "cascade": np.logspace(2, 7, 100+1),  # Default: same as global
        "track": np.logspace(2, 7, 100+1),    # Default: same as global
        "double_cascade": np.logspace(2, 7, 100+1)  # Default: same as global
    }
    
    line_style = "-"
    fig, ax = plt.subplots(figsize=(7, 5))
    morphology_dfs = {}  # Store dataframes for each morphology
    
    for morphology_index, morphology in enumerate(["cascade", "track", "double_cascade"]):
        color = cm((float(morphology_index) / float(3)) * 0.8 + 0.1)
        #label = f"All flavors {morphology}"
        label = f"Muons {morphology}"
        
        # Combine all flavors (use "nu" key which includes all neutrino types)
        # Or combine all 2nu flavors explicitly
        """particle_mask_all_flavors = np.logical_or.reduce([
            particle_masks["2nue"],
            particle_masks["2numu"],
            particle_masks["2nutau"]
        ])"""
        particle_mask_all_flavors = np.logical_or.reduce([
            particle_masks['mu'],
        ])
        
        # Apply morphology mask
        particle_mask = np.logical_and(
            particle_mask_all_flavors, morphology_masks[morphology_index]
        )
        
        # Get custom energy bins for this morphology
        custom_energy_bins = morphology_energy_bins[morphology]
        
        # Compute effective area with custom binning and get dataframe
        # Use factor=0.5 to match individual flavor files which compute (nu+nubar)/2 for each flavor
        # When summing individual flavors: (nu_e+nubar_e)/2 + (nu_mu+nubar_mu)/2 + (nu_tau+nubar_tau)/2
        # When combining all flavors first with factor=0.5: (nu_e+nubar_e+nu_mu+nubar_mu+nu_tau+nubar_tau)/2
        # These should be mathematically equivalent, so factor=0.5 is correct
        df = compute_effective_area_df(particle_mask, custom_energy_bins=custom_energy_bins, factor=0.5)
        morphology_dfs[morphology] = df
        
        # Optionally save dataframe to CSV
        # df.to_csv(outdir + f"/2nu_all_{morphology}_effective_area.csv", index=False)
        
        # Create energy masks for plotting
        nu_energy_mapping_custom = np.digitize(primaryEnergy, bins=custom_energy_bins) - 1
        custom_nu_energy_masks = [nu_energy_mapping_custom == i for i in range(len(custom_energy_bins) - 1)]
        masks = np.logical_and(particle_mask[None, :], custom_nu_energy_masks)
        
        # Plot the line (use factor=0.5 for consistency)
        plot_line(ax, masks, color, line_style, label, factor=0.5, energy_bins_plot=custom_energy_bins)
    
    #format_axis(ax)
    #save(fig, "effective_area_energy_all_flavors_morphology")
    
    # Return dataframes (or save them)
    return morphology_dfs


    """
    # Let's make an effective area vs. energy plot split by neutrino flavor
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        line_style = line_styles[0]
        label = f"ν_{flavor} + ν̄_{flavor}"
        particle_key = "2nu" + flavor
        particle_mask = particle_masks[particle_key]
        masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
        # The factor of 0.5 is needed so that we compute the average
        # neutrino/antineutrino effective area. This is in contrast to the
        # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
        # sum of the neutrino and antineutrino effective areas.
        plot_line(ax, masks, color, line_style, label, factor=0.5)
    format_axis(ax)
    save(fig, "effective_area_energy_2nu_flavor")
    
    # Make an effective area vs. energy plot split by neutrino type and morphology
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        #for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
        #    zip(["", "bar"], [False, True])
        #):
        line_style = "-"
        fig, ax = plt.subplots(figsize=(7, 5))
        for morphology_index, morphology in enumerate(["cascade", "track", "double_cascade"]):
            color = cm((float(morphology_index) / float(3)) * 0.8 + 0.1)
            flavor_str = flavor if flavor == 'e' else '\\' + flavor
            label = f"ν_{flavor_str} + ν̄_{flavor_str} {morphology}"
            particle_key = "2nu" + flavor
            particle_mask = particle_masks[particle_key]
            particle_mask = np.logical_and(
                particle_mask, morphology_masks[morphology_index]
            )
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            # The factor of 0.5 is needed so that we compute the average
            # neutrino/antineutrino effective area. This is in contrast to the
            # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
            # sum of the neutrino and antineutrino effective areas.
            plot_line(ax, masks, color, line_style, label, factor=0.5)
        format_axis(ax)
        save(
            fig,
            "effective_area_energy_nu"
            + flavor
            #+ ("bar" if is_antiparticle else "")
            + "_morphology",
        )
    """
    
    

    """# Make an effective area vs. energy plot split by neutrino type
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
            zip(["", "bar"], [False, True])
        ):
            line_style = line_styles[antiparticle_index]
            label = (
                f"ν_{flavor if flavor == 'e' else '\\' + flavor}"
                + (f"̄" if is_antiparticle else "")
            )
            particle_key = "nu" + flavor + antiparticle_suffix
            particle_mask = particle_masks[particle_key]
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            plot_line(ax, masks, color, line_style, label)
    format_axis(ax)
    save(fig, "effective_area_energy_nu_nubar_flavor")

    # Make an effective area vs. energy plot
    for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
        zip(["", "bar"], [False, True])
    ):
        fig, ax = plt.subplots(figsize=(7, 5))
        for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
            color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
            line_style = "-"
            label = (
                f"ν_{flavor if flavor == 'e' else '\\' + flavor}"
                + (f"̄" if is_antiparticle else "")
            )
            particle_key = "nu" + flavor + antiparticle_suffix
            particle_mask = particle_masks[particle_key]
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            plot_line(ax, masks, color, line_style, label)
        format_axis(ax)
        save(
            fig,
            "effective_area_energy_nu" + ("bar" if is_antiparticle else "") + "_flavor",
        )

    # Make an effective area vs. energy plot split by neutrino type and interaction type
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        for antiparticle_index, (antiparticle_suffix, is_antiparticle) in enumerate(
            zip(["", "bar"], [False, True])
        ):
            line_style = "-"
            fig, ax = plt.subplots(figsize=(7, 5))
            for interaction_index, interaction in enumerate(["CC", "NC", "GR"]):
                color = cm((float(interaction_index) / float(3)) * 0.8 + 0.1)
                label = (
                    f"ν_{flavor if flavor == 'e' else '\\' + flavor}"
                    + (f"̄" if is_antiparticle else "")
                    + f" {interaction}"
                )
                particle_key = "nu" + flavor + antiparticle_suffix
                particle_mask = particle_masks[particle_key]
                particle_mask = np.logical_and(
                    particle_mask, interaction_masks[interaction_index]
                )
                masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
                plot_line(ax, masks, color, line_style, label)
            format_axis(ax)
            save(
                fig,
                "effective_area_energy_nu"
                + flavor
                + ("bar" if is_antiparticle else ""),
            )

    # Make an effective area vs. energy plot split by flavor and interaction type
    fig, ax = plt.subplots(figsize=(7, 5))
    for flavor_index, flavor in enumerate(["e", "mu", "tau"]):
        color = cm((float(flavor_index) / float(n_flavors)) * 0.8 + 0.1)
        for interaction_index, interaction in enumerate(["CC"]):
            line_style = line_styles[interaction_index]
            label = (
                f"ν_{flavor if flavor == 'e' else '\\' + flavor}"
                + f" + ν̄_{flavor if flavor == 'e' else '\\' + flavor}"
                + f" {interaction}"
            )
            particle_key = "2nu" + flavor
            particle_mask = particle_masks[particle_key]
            particle_mask = np.logical_and(
                particle_mask, interaction_masks[interaction_index]
            )
            masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
            # The factor of 0.5 is needed so that we compute the average
            # neutrino/antineutrino effective area. This is in contrast to the
            # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
            # sum of the neutrino and antineutrino effective areas.
            plot_line(ax, masks, color, line_style, label, factor=0.5)

    color = "#4d4d4d"
    line_style = line_styles[1]
    label = "NC All Flavor"
    masks = np.logical_and(interaction_masks[1][None, :], nu_energy_masks)
    # The factor of 0.5 is needed so that we compute the average
    # neutrino/antineutrino effective area. This is in contrast to the
    # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
    # sum of the neutrino and antineutrino effective areas.
    plot_line(ax, masks, color, line_style, label, factor=0.5)

    line_style = line_styles[2]
    color = cm((float(0) / float(n_flavors)) * 0.8 + 0.1)
    flavor = "e"
    label = (
        f"ν_{flavor}"
        + f" + ν̄_{flavor}"
        + " GR"
    )
    particle_key = "2nu" + flavor
    particle_mask = particle_masks[particle_key]
    particle_mask = np.logical_and(particle_mask, interaction_masks[2])
    masks = np.logical_and(particle_mask[None, :], nu_energy_masks)
    # The factor of 0.5 is needed so that we compute the average
    # neutrino/antineutrino effective area. This is in contrast to the
    # effective area plot (FIG. 33) in PhysRevD.104.022002 which plots the
    # sum of the neutrino and antineutrino effective areas.
    plot_line(ax, masks, color, line_style, label, factor=0.5)

    format_axis(ax)
    save(fig, "effective_area_energy_2nu_flavor_interaction")"""


plot_effective_areas()
