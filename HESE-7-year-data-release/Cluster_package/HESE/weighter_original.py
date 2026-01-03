import numpy as np
import gc

import autodiff as ad
import det_sys_weights


class Weighter:
    def __init__(self, mc, model, nuSIprop=None):
        """
        Parameters
        ----------
        mc : structured array
            MC events
        model : str
            Astrophysical flux model: "spl", "cutoff", or "nusiprop"
        nuSIprop : object, optional
            Pre-initialized nuSIprop object (required when model="nusiprop").
            For other models this is ignored.
        """

        self.mc = mc
        # The SysWeighter class handles the weight corrections arising from varying detector
        # systematic parameters.
        self.sys_weighter = det_sys_weights.SysWeighter(self.mc)

        self.model = model.lower()
        self.nuSIprop = nuSIprop

    def flux_power_law(self, energy, norm, gamma, pivot):
        e_scale = energy / pivot
        spectrum = ad.pow_r(e_scale, ad.mul(gamma, -1.0))
        flux = ad.mul_grad(norm, spectrum)
        return flux

    def flux_spl(self, mc, astro_norm, astro_gamma, pivot_point=1e5):
        energy = mc["primaryEnergy"]
        flux = self.flux_power_law(energy, astro_norm, astro_gamma, pivot_point)

        # astro_norm is the 6 neutrino normalization so we need to convert it to the flux for 1 neutrino
        astro_flux = 1e-18 / 6.0
        flux = ad.mul(flux, astro_flux)

        return flux

    # ------------------------------
    # nuSIprop implementation
    # ------------------------------
    def _compute_nuSIprop_flux_scalar(self, energy, Mphi_val, g_val, si_val, mntot_val):
        """
        Helper function to compute nuSIprop flux for given scalar parameter values.
        Returns flux_total as a numpy array (no gradients).
        """
        if self.nuSIprop is None:
            raise ValueError("nuSIprop object must be provided when using model 'nusiprop'")

        norm_base = 1e-18  # Base normalization inside nuSIprop (astro_norm scales later)
        try:
            # nuSIprop expects mphi in eV, so convert from GeV to eV (×1e9),
            # keeping consistency with the existing implementation in weighter.py.
            self.nuSIprop.set_parameters(
                mphi=Mphi_val * 1e6,
                g=g_val,
                si=si_val,
                norm=norm_base,
                mntot=mntot_val,
            )
            self.nuSIprop.evolve()
            flux_el = self.nuSIprop.interp_flux_el(energy)
            flux_mu = self.nuSIprop.interp_flux_mu(energy)
            flux_ta = self.nuSIprop.interp_flux_ta(energy)
            flux_total = flux_el + flux_mu + flux_ta
            del flux_el, flux_mu, flux_ta
            gc.collect()
            
            # Check for NaN or inf values (can occur due to numerical overflow)
            # even if no exception was raised
            if np.any(~np.isfinite(flux_total)):
                # Return zeros if flux contains NaN/inf (will cause fit to return inf LLH)
                return np.zeros(len(energy))
            
            return flux_total
        except Exception as e:
            print(f"Warning: nuSIprop calculation failed: {e}")
            return np.zeros(len(energy))

    def nuSIprop_flux(self, mc, astro_norm, astro_gamma, Mphi, g, mntot):
        """
        Calculate nuSIprop flux with autodiff support, including finite-difference
        gradients for Mphi, g, mntot, and astro_gamma (used as the spectral index si).

        Parameters
        ----------
        mc : array
            MC events
        astro_norm : tuple
            Autodiff tuple [value, gradient] for astrophysical normalization
        astro_gamma : tuple
            Autodiff tuple [value, gradient] for astrophysical spectral index
            (used as si in nuSIprop)
        Mphi : tuple
            Autodiff tuple [value, gradient] for Mphi parameter
        g : tuple
            Autodiff tuple [value, gradient] for g parameter
        mntot : tuple
            Autodiff tuple [value, gradient] for mntot parameter

        Notes
        -----
        - Only parameters with non‑zero gradient components contribute to the
          finite-difference gradients; fixing a parameter in the fit (zeroing
          its gradient) automatically disables its finite-difference derivative.
        """
        # Extract scalar values from autodiff tuples
        Mphi_val = Mphi[0]
        g_val = g[0]
        mntot_val = mntot[0]
        si_val = astro_gamma[0]  # Use astro_gamma as spectral index (si)

        energy = mc["primaryEnergy"]
        # Convert to eV (as used in nuSIprop); energy is a regular array
        energy = energy * 1e9

        # Check for invalid energies
        """if np.any(~np.isfinite(energy)):
            print(
                f"Warning: Found {np.sum(~np.isfinite(energy))} invalid energy values "
                "(NaN or Inf) in MC; replacing with 1e14 eV"
            )
            energy = np.where(np.isfinite(energy), energy, 1e14)"""

        # Base flux (no astro_norm scaling yet)
        flux_total = self._compute_nuSIprop_flux_scalar(
            energy, Mphi_val, g_val, si_val, mntot_val
        )

        # Determine which global parameter indices correspond to which nuSIprop params
        n_params = len(astro_norm[1])
        astro_norm_idx = np.where(astro_norm[1] != 0)[0]
        astro_gamma_idx = np.where(astro_gamma[1] != 0)[0]
        Mphi_idx = np.where(Mphi[1] != 0)[0]
        g_idx = np.where(g[1] != 0)[0]
        mntot_idx = np.where(mntot[1] != 0)[0]

        # Initialize gradient for base flux (before astro_norm scaling)
        flux_grad = np.zeros((len(flux_total), n_params))

        # Relative finite-difference step
        # OBS! ÄNDRA FRÅN 1E-4 TILL 1E-3 FÖRSÄMRADE -LLH MED 10 PUNKTER SÅ TA INTE SÄMRE ÄN SÅ
        eps = 1e-3

        # Mphi gradient
        if len(Mphi_idx) > 0:
            Mphi_pert_up = Mphi_val * (1.0 + eps) if Mphi_val > 0 else Mphi_val + eps
            Mphi_pert_down = Mphi_val * (1.0 - eps) if Mphi_val > 0 else Mphi_val - eps
            flux_pert_up = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_pert_up, g_val, si_val, mntot_val
            )
            flux_pert_down = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_pert_down, g_val, si_val, mntot_val
            )
            flux_grad[:, Mphi_idx[0]] = (flux_pert_up - flux_pert_down) / (Mphi_pert_up - Mphi_pert_down)

        # g gradient
        if len(g_idx) > 0:
            g_pert_up = g_val * (1.0 + eps) if g_val > 0 else g_val + eps
            g_pert_down = g_val * (1.0 - eps) if g_val > 0 else g_val - eps
            flux_pert_up = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_pert_up, si_val, mntot_val
            )
            flux_pert_down = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_pert_down, si_val, mntot_val
            )
            flux_grad[:, g_idx[0]] = (flux_pert_up - flux_pert_down) / (g_pert_up - g_pert_down)

        # mntot gradient
        if len(mntot_idx) > 0:
            mntot_pert_up = mntot_val + eps *10 * max(abs(mntot_val), 0.01)  # Multiply by 10 due to flat direction
            mntot_pert_down = mntot_val - eps *10 * max(abs(mntot_val), 0.01)  # Multiply by 10 due to flat direction
            flux_pert_up = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_val, si_val, mntot_pert_up
            )
            flux_pert_down = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_val, si_val, mntot_pert_down
            )
            flux_grad[:, mntot_idx[0]] = (flux_pert_up - flux_pert_down) / (
                mntot_pert_up - mntot_pert_down
            )

        # astro_gamma (si) gradient
        if len(astro_gamma_idx) > 0:
            si_pert_up = si_val + eps * 0.1 * max(abs(si_val), 1.0)  # 
            si_pert_down = si_val - eps * 0.1* max(abs(si_val), 1.0)
            flux_pert_up = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_val, si_pert_up, mntot_val
            )
            flux_pert_down = self._compute_nuSIprop_flux_scalar(
                energy, Mphi_val, g_val, si_pert_down, mntot_val
            )
            flux_grad[:, astro_gamma_idx[0]] = (flux_pert_up - flux_pert_down) / (
                si_pert_up - si_pert_down
            )

        # Base flux as autodiff tuple (no astro_norm scaling yet)
        flux_base = (flux_total, flux_grad)

        # Apply astro_norm using autodiff; this adds its own gradient component(s)
        flux_with_norm = ad.mul_grad(astro_norm, flux_base)

        return flux_with_norm
    
    def flux_cutoff(self, mc, astro_norm, astro_gamma, cutoff_energy):
        energy = mc["primaryEnergy"]
        flux = self.flux_power_law(energy, astro_norm, astro_gamma, pivot=1e5)
        #flux = ad.mul(flux, ad.exp(ad.div_r(-energy, cutoff_energy)))
        e_factor = ad.div_r(-energy, cutoff_energy)
        e_term = ad.pow_r(np.e, e_factor)
        flux = ad.mul_grad(flux, e_term)
        
        # astro_norm is the 6 neutrino normalization so we need to convert it to the flux for 1 neutrino
        astro_flux = 1e-18 / 6.0
        flux = ad.mul(flux, astro_flux)
        
        return flux
    
    def flux_cutoff_old(self, mc, astro_norm, astro_gamma, cutoff_energy):
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
        #cutoff_energy = p["cutoff_energy"]

        # Optional nuSIprop physics parameters (only needed for model="nusiprop").
        # If they are not included in parameter_names/params, we raise an error
        # when the nuSIprop model is requested, but leave other models untouched.
        Mphi = p.get("Mphi", None)
        g = p.get("g", None)
        mntot = p.get("mntot", None)
        
        cutoff_energy = p.get("cutoff_energy", None)

        # Calculate the expected neutrino flux from each component
        if self.model == "spl":
            #print('Using SPL model')
            astro_fluxes = self.flux_spl(
                self.mc, astro_norm=astro_norm, astro_gamma=astro_gamma
            )
        elif self.model == "cutoff":
            #print('Using cutoff model')
            astro_fluxes = self.flux_cutoff(
                self.mc,
                astro_norm=astro_norm,
                astro_gamma=astro_gamma,
                cutoff_energy=cutoff_energy,
            )
        elif self.model == "nusiprop":
            #print('Using nuSIprop model')
            if self.nuSIprop is None:
                raise ValueError(
                    "nuSIprop model selected, but no nuSIprop object was provided "
                    "to Weighter(..., nuSIprop=<obj>)."
                )
            if Mphi is None or g is None or mntot is None:
                raise ValueError(
                    "nuSIprop model requires 'Mphi', 'g', and 'mntot' parameters "
                    "to be present in parameter_names."
                )
            astro_fluxes = self.nuSIprop_flux(
                self.mc,
                astro_norm=astro_norm,
                astro_gamma=astro_gamma,
                Mphi=Mphi,
                g=g,
                mntot=mntot,
            )
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

        neutrino_flux = ad.plus_grad(
            astro_weights, ad.plus_grad(conv_weights, prompt_weights)
        )

        neutrino_weights = ad.mul(neutrino_flux, self.mc["weightOverFluxOverLivetime"])

        weights = ad.plus_grad(neutrino_weights, muon_weights)

        weights = ad.mul(weights, livetime)

        return weights