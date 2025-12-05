import numpy as np
import gc

import autodiff as ad
import det_sys_weights

import sys
import os
import os.path
# Add nuSIprop to path (../../nuSIprop from this file's location)
base_path = os.path.dirname(os.path.abspath(__file__))
"""nuSIprop_path = os.path.abspath(os.path.join(base_path, '..', '..', 'nuSIprop'))
if nuSIprop_path not in sys.path:
    sys.path.insert(0, nuSIprop_path)
import nuSIprop"""


class Weighter:
    def __init__(self, mc, nuSIprop=False, model="spl", simple=False):

        self.mc = mc
        # The SysWeighter class handles the weight corrections arising from varying detector
        # systematic parameters.
        self.sys_weighter = det_sys_weights.SysWeighter(self.mc)
        
        if nuSIprop:
            self.nuSIprop = nuSIprop
        else:
            self.nuSIprop = None
        
        # Model selection: "spl", "lp", "cutoff", "nusiprop"
        self.model = model.lower()
        self.simple = simple
        # Model dispatch dictionary - maps model names to flux calculation methods
        # This makes it easy to add new models
        self.model_dispatch = {
            "spl": self._flux_spl,
            "lp": self._flux_lp,
            "cutoff": self._flux_cutoff,
            "nusiprop": self._flux_nusiprop,
        }
        

    def flux_power_law(self, energy, norm, gamma, pivot, cutoff=None, beta=None):
        e_scale = energy / pivot
        slope = gamma
        if beta is not None:
            # beta is a scalar autodiff tuple, log10_e_scale is a regular array
            # Use mul_r to multiply regular value by autodiff tuple
            log10_e_scale = np.log10(e_scale)
            beta_log_term = ad.mul_r(log10_e_scale, beta)
            slope = ad.plus_grad(gamma, beta_log_term)
        if cutoff is not None:
            spectrum = ad.mul_r(ad.pow_r(e_scale, ad.mul(slope, -1.0)), ad.less_than(e_scale, cutoff))
        else:
            spectrum = ad.pow_r(e_scale, ad.mul(slope, -1.0))
        flux = ad.mul_grad(norm, spectrum)
        return flux
    

    def flux_spl(self, mc, astro_norm, astro_gamma, pivot_point=1e5):
        energy = mc["primaryEnergy"]
        flux = self.flux_power_law(energy, astro_norm, astro_gamma, pivot_point)

        # astro_norm is the 6 neutrino normalization so we need to convert it to the flux for 1 neutrino
        astro_flux = 1e-18 / 6.0
        flux = ad.mul(flux, astro_flux)

        return flux
    
    def flux_cutoff(self, mc, astro_norm, astro_gamma, cutoff_energy):
        """
        Calculate flux with exponential cutoff: flux = norm * (E/E_pivot)^(-gamma) * exp(-E/E_cutoff)
        
        Parameters:
        -----------
        mc : array
            MC events
        astro_norm : tuple
            Autodiff tuple [value, gradient] for astro normalization parameter
        astro_gamma : tuple
            Autodiff tuple [value, gradient] for astro spectral index parameter
        cutoff_energy : tuple
            Autodiff tuple [value, gradient] for cutoff energy parameter
        """
        energy = mc["primaryEnergy"]
        
        # Calculate power law spectrum
        power_law_flux = self.flux_power_law(energy, astro_norm, astro_gamma, pivot=1e5)
        
        # Calculate exponential cutoff: exp(-energy/E_cutoff)
        # energy is a regular array, cutoff_energy is an autodiff tuple
        # Use div_r to divide regular array by autodiff tuple
        neg_energy_over_cutoff = ad.div_r(-energy, cutoff_energy)
        # Take exp of the result
        cutoff_factor = ad.exp(neg_energy_over_cutoff)
        
        # Multiply power law by cutoff factor
        flux = ad.mul_grad(power_law_flux, cutoff_factor)
        
        # astro_norm is the 6 neutrino normalization so we need to convert it to the flux for 1 neutrino
        astro_flux = 1e-18 / 6.0
        flux = ad.mul(flux, astro_flux)

        return flux
    
    def flux_lp(self, mc, astro_norm, astro_gamma, beta):
        energy = mc["primaryEnergy"]
        flux = self.flux_power_law(energy, astro_norm, astro_gamma, pivot=1e5, beta=beta)

        # astro_norm is the 6 neutrino normalization so we need to convert it to the flux for 1 neutrino
        astro_flux = 1e-18 / 6.0
        flux = ad.mul(flux, astro_flux)

        return flux
        
    def flux_nusiprop_simple(self, mc, astro_norm, astro_gamma):
        energy = mc["primaryEnergy"]
        energy = energy * 1e9
        self.nuSIprop.evolve()
        flux_el = self.nuSIprop.interp_flux_el(energy)
        flux_mu = self.nuSIprop.interp_flux_mu(energy)
        flux_ta = self.nuSIprop.interp_flux_ta(energy)
        flux_total = flux_el + flux_mu + flux_ta
        
        # Convert flux_total (NumPy array) to autodiff tuple with zero gradients
        # flux_total has no gradients (it's a constant from nuSIprop)
        n_params = len(astro_norm[1])  # Number of parameters
        flux_grad = np.zeros((len(flux_total), n_params))
        flux_tuple = (flux_total, flux_grad)
        
        # Multiply by astro_norm (which has gradients)
        #flux = ad.mul_grad(astro_norm, flux_tuple)
        
        # Apply normalization factor
        norm_factor = 1e-18 * 1.9
        flux = ad.mul(flux_tuple, norm_factor)

        return flux
        
    def _compute_nuSIprop_flux_scalar(self, energy, Mphi_val, g_val, si_val, mntot_val):
        """
        Helper function to compute nuSIprop flux for given scalar parameter values.
        Returns flux_total as a numpy array.
        """
        norm_base = 1e-18  # Base normalization
        try:
            self.nuSIprop.set_parameters(mphi=Mphi_val*1e6, g=g_val, si=si_val, norm=norm_base, mntot=mntot_val)
            self.nuSIprop.evolve()
            flux_el = self.nuSIprop.interp_flux_el(energy)
            flux_mu = self.nuSIprop.interp_flux_mu(energy)
            flux_ta = self.nuSIprop.interp_flux_ta(energy)
            flux_total = flux_el + flux_mu + flux_ta
            del flux_el, flux_mu, flux_ta
            gc.collect()
            return flux_total
        except Exception as e:
            print(f"Warning: nuSIprop calculation failed: {e}")
            return np.zeros(len(energy))
    
    def nuSIprop_flux(self, mc, astro_norm, astro_gamma, Mphi, g, mntot):
        """
        Calculate nuSIprop flux with autodiff support, including finite-difference
        gradients for Mphi, g, mntot, and astro_gamma.
        
        Parameters:
        -----------
        mc : array
            MC events
        astro_norm : tuple
            Autodiff tuple [value, gradient] for astro normalization parameter
        astro_gamma : tuple
            Autodiff tuple [value, gradient] for astro spectral index parameter
        Mphi : tuple
            Autodiff tuple [value, gradient] for Mphi parameter
        g : tuple
            Autodiff tuple [value, gradient] for g parameter
        mntot : tuple
            Autodiff tuple [value, gradient] for mntot parameter
        """
        # Extract scalar values from autodiff tuples
        Mphi_val = Mphi[0]
        g_val = g[0]
        mntot_val = mntot[0]
        si_val = astro_gamma[0]  # Use astro_gamma as spectral index
        
        energy = mc["primaryEnergy"]
        energy = energy * 1e9  # Convert to eV (regular numpy multiplication, not autodiff)
        
        
        # Check for invalid energy values
        if np.any(~np.isfinite(energy)):
            print(f"Warning: Found {np.sum(~np.isfinite(energy))} invalid energy values (NaN or Inf)")
            energy = np.where(np.isfinite(energy), energy, 1e14)  # Replace invalid with 10^14 eV
        
        # Compute flux at current parameter values
        flux_total = self._compute_nuSIprop_flux_scalar(energy, Mphi_val, g_val, si_val, mntot_val)
        
        # Get parameter indices from gradient arrays
        n_params = len(astro_norm[1])  # Total number of parameters
        astro_norm_idx = np.where(astro_norm[1] != 0)[0]
        astro_gamma_idx = np.where(astro_gamma[1] != 0)[0]
        Mphi_idx = np.where(Mphi[1] != 0)[0]
        g_idx = np.where(g[1] != 0)[0]
        mntot_idx = np.where(mntot[1] != 0)[0]
        
        # Initialize gradient array
        flux_grad = np.zeros((len(flux_total), n_params))
        
        # Finite difference step size (relative perturbation)
        # Use a small but not too small step to avoid numerical issues
        eps = 1e-4  # Slightly larger step for better numerical stability
        
        # Compute finite-difference gradient for Mphi
        # Store gradient of flux_base (before multiplying by astro_norm)
        if len(Mphi_idx) > 0:
            Mphi_pert = Mphi_val * (1.0 + eps) if Mphi_val > 0 else Mphi_val + eps
            flux_pert = self._compute_nuSIprop_flux_scalar(energy, Mphi_pert, g_val, si_val, mntot_val)
            flux_grad[:, Mphi_idx[0]] = (flux_pert - flux_total) / (Mphi_pert - Mphi_val)
        
        # Compute finite-difference gradient for g
        if len(g_idx) > 0:
            g_pert = g_val * (1.0 + eps) if g_val > 0 else g_val + eps
            flux_pert = self._compute_nuSIprop_flux_scalar(energy, Mphi_val, g_pert, si_val, mntot_val)
            flux_grad[:, g_idx[0]] = (flux_pert - flux_total) / (g_pert - g_val)
        
        # Compute finite-difference gradient for mntot
        if len(mntot_idx) > 0:
            mntot_pert = mntot_val + eps * max(abs(mntot_val), 0.01)
            flux_pert = self._compute_nuSIprop_flux_scalar(energy, Mphi_val, g_val, si_val, mntot_pert)
            flux_grad[:, mntot_idx[0]] = (flux_pert - flux_total) / (mntot_pert - mntot_val)
        
        # Compute finite-difference gradient for astro_gamma (si)
        if len(astro_gamma_idx) > 0:
            si_pert = si_val + eps * max(abs(si_val), 0.1)
            flux_pert = self._compute_nuSIprop_flux_scalar(energy, Mphi_val, g_val, si_pert, mntot_val)
            flux_grad[:, astro_gamma_idx[0]] = (flux_pert - flux_total) / (si_pert - si_val)
        
        # Create flux_base with gradients for nuSIprop parameters
        # flux_base represents the flux before multiplying by astro_norm
        flux_base = (flux_total, flux_grad)
        
        # Multiply by astro_norm using autodiff
        # This computes: flux_final = astro_norm * flux_base
        # And gradients: d(flux_final)/d(param) = astro_norm * d(flux_base)/d(param) + flux_base * d(astro_norm)/d(param)
        # The autodiff library handles the scaling automatically
        flux_with_norm = ad.mul_grad(astro_norm, flux_base)
        
        return flux_with_norm
    
    # Model-specific flux calculation methods (called via dispatch)
    def _flux_spl(self, p, astro_norm, astro_gamma):
        """Single power law model"""
        return self.flux_spl(self.mc, astro_norm=astro_norm, astro_gamma=astro_gamma)
    
    def _flux_lp(self, p, astro_norm, astro_gamma):
        """Log-parabola model"""
        if "beta" not in p:
            raise ValueError("Log-parabola model requires 'beta' parameter")
        return self.flux_lp(self.mc, astro_norm=astro_norm, astro_gamma=astro_gamma, beta=p["beta"])
    
    def _flux_cutoff(self, p, astro_norm, astro_gamma):
        """Exponential cutoff model"""
        if "cutoff_energy" not in p:
            raise ValueError("Cutoff model requires 'cutoff_energy' parameter")
        cutoff_energy = p["cutoff_energy"]
        return self.flux_cutoff(self.mc, astro_norm=astro_norm, astro_gamma=astro_gamma, cutoff_energy=cutoff_energy)
    
    def _flux_nusiprop(self, p, astro_norm, astro_gamma):
        """nuSIprop model"""
        if not self.nuSIprop:
            raise ValueError("nuSIprop model requires nuSIprop object to be provided")
        if "Mphi" not in p or "g" not in p or "mntot" not in p:
            raise ValueError("nuSIprop model requires 'Mphi', 'g', and 'mntot' parameters")
        Mphi = p["Mphi"]
        g = p["g"]
        mntot = p["mntot"]
        # For nuSIprop, astro_gamma is used as the spectral index (si)
        return self.nuSIprop_flux(self.mc, astro_norm, astro_gamma, Mphi, g, mntot, simple=self.simple)

    def weight_nunubar_ratio(self, mc, nunubar_ratio):
        p_id = mc["primaryType"]

        weights = np.empty(len(mc))
        gradients = np.empty((len(mc), len(nunubar_ratio[1])))

        weights[p_id > 0], gradients[p_id > 0] = nunubar_ratio
        weights[p_id < 0], gradients[p_id < 0] = ad.minus_r(2.0, nunubar_ratio)

        return weights, gradients

    def flux_conv(
        self,
        mc,
        conv_norm,
        kpi_ratio,
        cr_delta_gamma,
        nunubar_ratio,
        pivot_point=2020.0,
    ):
        energy = mc["primaryEnergy"]
        tilt_flux = self.flux_power_law(energy, conv_norm, cr_delta_gamma, pivot_point)

        pion_flux = mc["pionFlux"]
        kaon_flux = mc["kaonFlux"]

        total_flux = ad.plus_r(pion_flux, ad.mul_r(kaon_flux, kpi_ratio))

        flux = ad.mul_grad(total_flux, tilt_flux)
        flux = ad.mul_grad(flux, self.weight_nunubar_ratio(mc, nunubar_ratio))

        return flux

    def flux_prompt(
        self,
        mc,
        prompt_norm,
        cr_delta_gamma,
        nunubar_ratio,
        pivot_point=7887.0,
    ):
        energy = mc["primaryEnergy"]
        flux = self.flux_power_law(energy, prompt_norm, cr_delta_gamma, pivot_point)

        prompt_flux = mc["promptFlux"]

        flux = ad.mul(flux, prompt_flux)
        flux = ad.mul_grad(flux, self.weight_nunubar_ratio(mc, nunubar_ratio))

        return flux

    def weight_muon(self, mc, muon_norm):

        return ad.mul(muon_norm, mc["muonWeightOverLivetime"])

    def astro_detector_correction(self, epsilon_dom, epsilon_head_on, anisotropy_scale):

        hole_ice_weight = self.sys_weighter.get_hole_ice_weights(
            "Astro", epsilon_head_on
        )
        dom_eff_weight = self.sys_weighter.get_dom_eff_weights("Astro", epsilon_dom)
        anisotropy_weight = self.sys_weighter.get_anisotropy_weights(
            "Astro", anisotropy_scale
        )

        return ad.mul_grad(
            ad.mul_grad(hole_ice_weight, dom_eff_weight), anisotropy_weight
        )

    def conv_detector_correction(self, epsilon_dom, epsilon_head_on, anisotropy_scale):

        hole_ice_weight = self.sys_weighter.get_hole_ice_weights(
            "Conv", epsilon_head_on
        )
        dom_eff_weight = self.sys_weighter.get_dom_eff_weights("Conv", epsilon_dom)
        anisotropy_weight = self.sys_weighter.get_anisotropy_weights(
            "Conv", anisotropy_scale
        )

        return ad.mul_grad(
            ad.mul_grad(hole_ice_weight, dom_eff_weight), anisotropy_weight
        )

    def prompt_detector_correction(
        self, epsilon_dom, epsilon_head_on, anisotropy_scale
    ):

        hole_ice_weight = self.sys_weighter.get_hole_ice_weights(
            "Prompt", epsilon_head_on
        )
        dom_eff_weight = self.sys_weighter.get_dom_eff_weights("Prompt", epsilon_dom)
        anisotropy_weight = self.sys_weighter.get_anisotropy_weights(
            "Prompt", anisotropy_scale
        )

        return ad.mul_grad(
            ad.mul_grad(hole_ice_weight, dom_eff_weight), anisotropy_weight
        )

    def get_weights(self, livetime, parameter_names, params):
        """
        Return
        ---------
        tuple
            Zeroth element contains the list of weights
            First element contains the list of the gradients for each weight
        """

        n_params = len(params)

        p = dict()

        # Initialize parameter vector with gradient
        for i, (name, param) in enumerate(zip(parameter_names, params)):
            p_grad = np.zeros(shape=n_params).astype(float)
            p_grad[i] = 1.0
            p[name] = [param, p_grad]

        # Each element is a tuple. The zeroth element is the value, and the
        # first element is the corresponding gradient
        astro_norm = p["astro_norm"]
        astro_gamma = p["astro_gamma"]
        conv_norm = p["conv_norm"]
        prompt_norm = p["prompt_norm"]
        kpi_ratio = p["kpi_ratio"]
        cr_delta_gamma = p["cr_delta_gamma"]
        epsilon_dom = p["epsilon_dom"]
        epsilon_head_on = p["epsilon_head_on"]
        anisotropy_scale = p["anisotropy_scale"]
        nunubar_ratio = p["nunubar_ratio"]
        muon_norm = p["muon_norm"]
        
        # nuSIprop parameters (fixed physics parameters, not fit parameters)
        # Note: si and norm are now taken from astro_gamma and astro_norm
        #Mphi = 7.5*1e6
        #g = 0.1
        #mntot = 0.07
        #nuSIprop_params = (Mphi, g, None, None, mntot)  # si and norm are placeholders

        # Calculate the expected neutrino flux from each component
        # Use model dispatch to select the appropriate flux calculation
        if self.model not in self.model_dispatch:
            raise ValueError(f"Unknown model: {self.model}. Available models: {list(self.model_dispatch.keys())}")
        """astro_fluxes = self.flux_spl(
            self.mc, astro_norm=astro_norm, astro_gamma=astro_gamma
        )"""
        flux_func = self.model_dispatch[self.model]
        if self.model == "nusiprop":
            if self.simple:
                astro_fluxes = self.flux_nusiprop_simple(self.mc, astro_norm, astro_gamma)
            else:
                astro_fluxes = self.nuSIprop_flux(self.mc, astro_norm, astro_gamma)
        else:
            astro_fluxes = flux_func(self.mc, astro_norm, astro_gamma)
        
        conv_fluxes = self.flux_conv(
            self.mc,
            conv_norm=conv_norm,
            kpi_ratio=kpi_ratio,
            cr_delta_gamma=cr_delta_gamma,
            nunubar_ratio=nunubar_ratio,
        )
        prompt_fluxes = self.flux_prompt(
            self.mc,
            prompt_norm=prompt_norm,
            cr_delta_gamma=cr_delta_gamma,
            nunubar_ratio=nunubar_ratio,
        )

        # Calculate the muon weights
        muon_weights = self.weight_muon(self.mc, muon_norm=muon_norm)
        #print('muon_weights', muon_weights)

        # Correct atmospheric weights with self-veto
        conv_fluxes = ad.mul(conv_fluxes, self.mc["conventionalSelfVetoCorrection"])
        prompt_fluxes = ad.mul(prompt_fluxes, self.mc["promptSelfVetoCorrection"])

        # Weight modifications due to detector systematics
        astro_weights = ad.mul_grad(
            astro_fluxes,
            self.astro_detector_correction(
                epsilon_dom, epsilon_head_on, anisotropy_scale
            ),
        )
        conv_weights = ad.mul_grad(
            conv_fluxes,
            self.conv_detector_correction(
                epsilon_dom, epsilon_head_on, anisotropy_scale
            ),
        )
        prompt_weights = ad.mul_grad(
            prompt_fluxes,
            self.prompt_detector_correction(
                epsilon_dom, epsilon_head_on, anisotropy_scale
            ),
        )
        #print('astro_weights', astro_weights)
        #print('conv_weights', conv_weights)
        #print('prompt_weights', prompt_weights)

        neutrino_flux = ad.plus_grad(
            astro_weights, ad.plus_grad(conv_weights, prompt_weights)
        )

        neutrino_weights = ad.mul(neutrino_flux, self.mc["weightOverFluxOverLivetime"])

        weights = ad.plus_grad(neutrino_weights, muon_weights)

        weights = ad.mul(weights, livetime)

        return weights
