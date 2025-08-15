import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad
import mcfit

from scipy.special import hyp2f1
from scipy.optimize import fsolve

from lp_utils.utils import SPEED_OF_LIGHT, read_json, read_pk
from lp_utils.filters_et_functions import top_hat_filter, wgc, j0, j1, d_dr_j0_of_kr


class Cosmology:
    """
    A class for handling cosmological calculations and parameters.
    This class allows for the initialization of cosmological parameters either via a preset or by manual specification.
    It provides methods for computing cosmological quantities such as comoving distances, growth factors, and cosmological volumes.

    Parameters
    ----------
    preset : str, optional
        Name of the preset cosmology to use. If provided, loads parameters from a corresponding JSON file.
    Omega_r : float, optional
        Radiation density parameter.
    Omega_m : float, optional
        Matter density parameter.
    Omega_DE : float, optional
        Dark energy density parameter.
    Omega_k : float, optional
        Curvature density parameter.
    h : float, optional
        Dimensionless Hubble parameter.
    sigma8 : float, optional
        RMS mass fluctuation amplitude at 8 Mpc/h.
    Pk_filename : str or None
        Filename for the power spectrum (if available in preset).

    Attributes
    ----------
    Omega_r : float
        Radiation density parameter.
    Omega_m : float
        Matter density parameter.
    Omega_DE : float
        Dark energy density parameter.
    Omega_L : float
        Alias for Omega_DE.
    Omega_k : float
        Curvature density parameter.
    h : float
        Dimensionless Hubble parameter.
    sigma8 : float
        RMS mass fluctuation amplitude at 8 Mpc/h.
    n_s : float
        Scalar spectral index (if available in preset).
    w : float
        Dark energy equation of state parameter (if available in preset).
    As : float
        Scalar amplitude (if available in preset).
    k : ndarray
        Wavenumber array for the power spectrum (if Pk_filename is provided).
    Pk : ndarray
        Power spectrum values from the file (if Pk_filename is provided).
    P : ndarray
        Power spectrum values, possibly rescaled to match sigma8 (if Pk_filename is provided).

    Methods
    -------
    choose_cosmo(cosmology)
        Loads cosmological parameters from a preset JSON file.
    E_late_times(z)
        Computes the dimensionless Hubble parameter E(z) for late times (ignoring radiation).
    E_correct(z)
        Computes the dimensionless Hubble parameter E(z) including radiation.
    comoving_distance(z, h_units=True)
        Calculates the comoving distance to redshift z, optionally in units of Mpc/h.
    comoving_distance_late_times(z, h_units=True)
        Calculates the comoving distance to redshift z using late-time approximation.
    comoving_distance_interp(use_late_times=False, z_vals=None)
        Returns an interpolator for comoving distance as a function of redshift.
    growth_factor(z_input)
        Computes the linear growth factor D(z) at given redshift(s).
    d_dz_growth_factor(z_input)
        Computes the derivative of the linear growth factor D(z) with respect to redshift.
    growth_rate(z_input)
        Computes the linear growth rate f(z) at given redshift(s).
    volume_zbin(zi, zf, fsky=None, solid_angle=None, use_late_times=False, z_vals=None)
        Computes the comoving volume between two redshifts for a given sky fraction or solid angle.
    get_vol_interp(zmin=0, zmax=2.5, fsky=None, solid_angle=None, z_vals=None)
        Returns an interpolator for the comoving volume as a function of redshift.
    find_z_for_target_volume(volume_target, fsky, z_min=0, z_max=2, z_vals=None)
        Finds the redshift corresponding to a target comoving volume.
    """

    def __init__(
        self,
        preset=None,
        Omega_r=None,
        Omega_m=None,
        Omega_DE=None,
        Omega_k=None,
        h=None,
        sigma8=None,
        Pk_filename=None,
    ):
        if preset is not None:
            self.cosmo = self.choose_cosmo(preset)
            self.h = self.cosmo["h"]
            self.Omega_r = self.cosmo["Omega_r"]
            self.Omega_m = self.cosmo["Omega_m"]
            self.Omega_L = self.cosmo["Omega_DE"]
            self.Omega_DE = self.cosmo["Omega_DE"]
            self.Omega_k = self.cosmo["Omega_k"]
            self.sigma8 = self.cosmo["sigma8"]
            self.n_s = self.cosmo["n_s"]
            self.w = self.cosmo.get("w", -1.0)
            self.As = self.cosmo.get("As")
            self.Pk_filename = self.cosmo.get("Pk_filename", None)
        else:
            if None in (Omega_r, Omega_m, Omega_DE, Omega_k, h, sigma8):
                raise ValueError(
                    "Must provide all cosmological parameters or use a preset"
                )
            self.Omega_r = Omega_r
            self.Omega_m = Omega_m
            self.Omega_DE = Omega_DE
            self.Omega_L = Omega_DE
            self.Omega_k = Omega_k
            self.sigma8 = sigma8
            self.Pk_filename = Pk_filename

        self.set_pk()

    def choose_cosmo(self, cosmology):
        """
        Loads cosmological parameters from a preset JSON file.

        Parameters
        ----------
        cosmology : str
            Name of the preset cosmology.

        Returns
        -------
        cosmo_dict : dict
            Dictionary of cosmological parameters.
        """
        list_of_cosmologies = [
            "raygal",
            "istf",
            "wmap1",
            "wmap3",
            "wmap5",
            "wmap7",
            "wmap9",
        ]
        if cosmology.lower() in list_of_cosmologies:
            cosmo_dict = self._read_cosmo_file(f"{cosmology}_cosmology.json")
        else:
            raise ValueError(
                f"Unknown preset cosmology: {cosmology}. Available presets are: {', '.join(list_of_cosmologies)}"
            )
        return cosmo_dict

    def _read_cosmo_file(self, filename):
        """
        Reads cosmological parameters from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file.

        Returns
        -------
        cosmo_dict : dict
            Dictionary of cosmological parameters.
        """
        cosmo_dict = read_json(filename)

        # For istf, calculate Omega_r if not provided
        if "istf" in filename and "Omega_r" not in cosmo_dict:
            cosmo_dict["Omega_r"] = (
                1.0
                - cosmo_dict["Omega_m"]
                - cosmo_dict["Omega_k"]
                - cosmo_dict["Omega_DE"]
            )

        return cosmo_dict

    def set_pk(self):
        """
        Sets the power spectrum based on the provided Pk_filename.
        """
        if self.Pk_filename is not None:
            print(f"Loading power spectrum from {self.Pk_filename}")
            self.k, self.Pk = read_pk(self.Pk_filename)
            sigma8_computed = compute_sigma8(self.k, self.Pk)
            if np.isclose(sigma8_computed, self.sigma8, rtol=1e-6):
                self.P = self.Pk
            else:
                print(
                    f"Warning: Computed sigma8 ({sigma8_computed}) does not match provided sigma8 ({self.sigma8}). "
                    "Rescaling the power spectrum to match the provided sigma8."
                )
                self.P = change_sigma8(self.k, self.Pk, self.sigma8)
                print("--------- After rescaling ---------")
                print(f"sigma8 = {self.sigma8:.8f}")
                print(f"sigma8 from Pk = {compute_sigma8(self.k, self.P):.8f}")

    def E_late_times(self, z):
        """
        Calculate the dimensionless Hubble parameter E(z) at late times.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to evaluate E(z).

        Returns
        -------
        float or ndarray
            The value(s) of the dimensionless Hubble parameter E(z) at the given redshift(s),
            computed as sqrt(Omega_m * (1 + z)^3 + Omega_k * (1 + z)^2 + Omega_L).
        """
        return np.sqrt(
            self.Omega_m * (1 + z) ** 3 + self.Omega_k * (1 + z) ** 2 + self.Omega_L
        )

    def E_correct(self, z):
        """
        Calculate the dimensionless Hubble parameter E(z) at redshift z using the corrected cosmological parameters.

        Parameters
        ----------
        z : float or array-like
            Redshift value(s) at which to evaluate E(z).

        Returns
        -------
        float or ndarray
            The value(s) of the dimensionless Hubble parameter E(z).

        Notes
        -----
        E(z) is computed as:
            E(z) = sqrt(
                Omega_r * (1 + z)^4 +
                Omega_m * (1 + z)^3 +
                Omega_k * (1 + z)^2 +
                Omega_L
        where Omega_r, Omega_m, Omega_k, and Omega_L are the density parameters for radiation, matter, curvature, and dark energy, respectively.
        """
        return np.sqrt(
            self.Omega_r * (1 + z) ** 4
            + self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_L
        )

    def comoving_distance(self, z, h_units=True):
        """
        Calculate the comoving distance to a given redshift z.

        Parameters
        ----------
        z : float or array-like
            Redshift(s) at which to compute the comoving distance.
        h_units : bool, optional
            If True (default), return the distance in units Mpc/h.

        Returns
        -------
        float or ndarray
            Comoving distance(s) corresponding to the input redshift(s).
        """
        h = 1.0
        if not h_units:
            h = self.h
        Dh = SPEED_OF_LIGHT * 0.01 / h
        z = np.asarray(z)
        if z.ndim == 0:
            integral = quad(lambda x: 1.0 / self.E_correct(x), 0.0, z)[0]
        else:
            integral = np.array(
                [quad(lambda x: 1.0 / self.E_correct(x), 0.0, z_i)[0] for z_i in z]
            )
        return Dh * integral

    def comoving_distance_late_times(self, z, h_units=True):
        """
        Calculate the comoving distance at late times for a given redshift.

        Parameters
        ----------
        z : float
            The redshift at which to compute the comoving distance.
        h_units : bool, optional
            If True (default), the result is returned in units of Mpc/h.

        Returns
        -------
        float
            The comoving distance to redshift z in Mpc.
        """
        h = 1.0
        if not h_units:
            h = self.h
        Dh = SPEED_OF_LIGHT * 0.01 / h
        integral = quad(lambda x: 1.0 / self.E_late_times(x), 0.0, z)
        return Dh * integral[0]

    def comoving_distance_interp(self, use_late_times=False, z_vals=None):
        """
        Returns a cubic interpolation function for the comoving distance as a function of redshift.

        Parameters
        ----------
        use_late_times : bool, optional
            If True, use `comoving_distance_late_times` for distance calculation.
            If False (default), use `comoving_distance`.
        z_vals : array-like or None, optional
            Array of redshift values at which to compute the comoving distance.
            If None, defaults to a linearly spaced array from 0.0 to 2.5 with 4000 points.

        Returns
        -------
        distance_cubic_interp : callable
            Cubic interpolation function mapping redshift to comoving distance.
        """
        if z_vals is None:
            z_vals = np.linspace(0.0, 2.5, 4000)
        distance_func = (
            self.comoving_distance_late_times
            if use_late_times
            else self.comoving_distance
        )
        dist_vals = np.array([distance_func(z) for z in z_vals])
        distance_cubic_interp = interp1d(z_vals, dist_vals, kind="cubic")
        return distance_cubic_interp

    def growth_factor(self, z_input):
        """
        Wrapper for _growth_factor_impl. See _growth_factor_impl docstring for details.
        """
        return _growth_factor_impl(z_input, self.Omega_m)

    def d_dz_growth_factor(self, z_input):
        """
        Wrapper for _d_dz_growth_factor_impl. See _d_dz_growth_factor_impl docstring for details.
        """
        return _d_dz_growth_factor_impl(z_input, self.Omega_m)

    def growth_rate(self, z_input):
        """
        Wrapper for _growth_rate_impl. See _growth_rate_impl docstring for details.
        """
        return _growth_rate_impl(z_input, self.Omega_m)

    def volume_zbin(
        self, zi, zf, fsky=None, solid_angle=None, use_late_times=False, z_vals=None
    ):
        """
        Calculates the comoving volume between two redshifts (zi and zf) within a specified sky area.

        Parameters
        ----------
        zi : float
            Initial redshift.
        zf : float
            Final redshift.
        fsky : float, optional
            Fraction of the sky covered (0 < fsky <= 1). If provided, solid_angle is ignored.
        solid_angle : float, optional
            Solid angle in steradians. Used if fsky is not provided.
        use_late_times : bool, optional
            If True, us comoving_distance_late_times to compute distances.
        z_vals : array-like, optional
            Redshift values for interpolation.

        Returns
        -------
        float
            Comoving volume between zi and zf within the specified sky area.

        Raises
        ------
        ValueError
            If neither fsky nor solid_angle is provided.
        """
        r = self.comoving_distance_interp(use_late_times, z_vals)
        if fsky is not None:
            omega = 4 * np.pi * fsky
        elif solid_angle is not None:
            omega = solid_angle
        else:
            raise ValueError("Either fsky or solid_angle must be provided")
        return omega * (r(zf) ** 3 - r(zi) ** 3) / 3

    def get_vol_interp(
        self, zmin=0, zmax=2.5, fsky=None, solid_angle=None, z_vals=None
    ):
        """
        Creates an interpolator function for the comoving volume between redshift bins.

        Parameters
        ----------
        zmin : float, optional
            Minimum redshift value for the volume calculation (default is 0).
        zmax : float, optional
            Maximum redshift value for the volume calculation (default is 2.5).
        fsky : float, optional
            Fraction of the sky covered (default is None). If not provided, it will be calculated from `solid_angle` if available.
        solid_angle : float, optional
            Solid angle in steradians (default is None). Used to compute `fsky` if provided.
        z_vals : array-like, optional
            Array of redshift values over which to compute the volume grid (default is None, which uses 4000 points between 0 and 2.5).

        Returns
        -------
        volume_interp : scipy.interpolate.interp1d
            Interpolator function that returns the comoving volume for a given redshift.
        """
        # Create interpolator for volume_zbin function
        if z_vals is None:
            z_vals = np.linspace(0.0, 2.5, 4000)
        if solid_angle is not None:
            fsky = solid_angle / (4 * np.pi)

        vol_grid = np.array(
            [self.volume_zbin(zmin, zmax, fsky) for zmax in z_vals]
        )  # Calculate volumes

        volume_interp = interp1d(z_vals, vol_grid, kind="cubic")
        return volume_interp

    def find_z_for_target_volume(
        self, volume_target, fsky, z_min=0, z_max=2, z_vals=None
    ):
        """
        Finds the redshift value `z` such that the comoving volume at that redshift matches the sepcified volume_target.

        Parameters
        ----------
        volume_target : float
            The target comoving volume to match.
        fsky : float
            Fraction of the sky covered (0 < fsky <= 1).
        z_min : float, optional
            Minimum redshift value to consider (default is 0).
        z_max : float, optional
            Maximum redshift value to consider (default is 2).
        z_vals : array-like, optional
            Optional array of redshift values for interpolation.

        Returns
        -------
        float
            The redshift value `z` such that the comoving volume at that redshift matches the sepcified volume_target.
        """
        return fsolve(
            lambda z: self.volume_zbin(z, z_max, fsky=fsky, z_vals=z_vals)
            - volume_target,
            (z_max + z_min) / 2,
        )[0]

    def Pk2xi(self, s_arr, z=0):
        def integrand(x, Px, r, xpiv=1):
            return x**2 * Px / (2.0 * np.pi) ** 3 * j0(x * r) * wgc(x, xpiv, 4)

        s_arr = np.asarray(s_arr)
        xi = []
        for si in s_arr:
            xi.append(integrate.simpson(integrand(self.k, self.P, si), self.k))
        xi = 4 * np.pi * self.growth_factor(z) ** 2 * np.array(xi)
        return xi

    def Pk2xiNL(self, s_arr, z=0, *args, **kwargs):
        def integrand(x, Px, r, xpiv=1):
            return (
                x**2
                * self.coefficient_form(z, *args, **kwargs)
                * Px
                / (2.0 * np.pi) ** 3
                * j0(x * r)
                * wgc(x, xpiv, 4)
            )

        s_arr = np.asarray(s_arr)
        xi = []
        for si in s_arr:
            xi.append(integrate.simpson(integrand(self.k, self.P, si), self.k))
        xi = 4 * np.pi * self.growth_factor(z) ** 2 * np.array(xi)
        return xi

    def binned_xiNL(self, s_arr, delta_r, z=0, *args, **kwargs):
        def integrand(x, Px, r, xpiv=1):
            return (
                x**2
                * self.coefficient_form(z, *args, **kwargs)
                * Px
                / (2.0 * np.pi) ** 3
                * j0_bar(x, r, delta_r)
                * wgc(x, xpiv, 4)
            )

        s_arr = np.asarray(s_arr)
        xi = []
        for si in s_arr:
            xi.append(integrate.simpson(integrand(self.k, self.P, si), self.k))
        xi = 4 * np.pi * self.growth_factor(z) ** 2 * np.array(xi)
        return xi

    def Pk2d_xi_ds(self, s_arr, z=0):
        """
        Computes the derivative of the linear two-point correlation function xi(s) with respect to s, given a power spectrum P(k) and an array of separations s.

        Parameters
        ----------
        s_arr : array_like    Array of separations at which to compute the derivative of xi.
        z : float, optional   Redshift at which to evaluate the growth factor (default is 0).

        Returns
        -------
        d_xi_ds : ndarray    The derivative of the two-point correlation function xi(s) with respect to s.
        """

        s_arr = np.asarray(s_arr)

        def integrand(x, Px_interp, r):
            return (
                x**2
                * Px_interp(x)
                / (2.0 * np.pi) ** 3
                * d_dr_j0_of_kr(x, r)
                * wgc(x, 1, 4)
            )

        Px_interp = interp1d(
            self.k, self.P, kind="cubic", bounds_error=False, fill_value=0.0
        )
        dxi = []
        for si in s_arr:
            val, _ = quad(lambda x: integrand(x, Px_interp, si), 0.001, 100, limit=200)
            dxi.append(val)
        dxi = 4 * np.pi * self.growth_factor(z) ** 2 * np.array(dxi)

        return dxi

    def dxi_ds(self, s_arr, z=0):
        return np.gradient(self.Pk2xi(s_arr, z), s_arr, edge_order=2)

    def dxiNL_ds(self, s_arr, z=0, *args, **kwargs):
        return np.gradient(self.Pk2xiNL(s_arr, z, *args, **kwargs), s_arr, edge_order=2)

    def coefficient_form(self, z, *args, **kwargs):
        if len(args) >= 1:
            b10 = args[0]
        else:
            b10 = kwargs.get("b10", 1.0)

        sigma0sq = self.sigma0sq(z, *args, **kwargs)
        Asq = self.Asq(z, b10)

        return Asq * np.exp(-(self.k**2) * sigma0sq)

    def Asq(self, z, b10=1.0):
        f = self.growth_rate(z)
        return b10**2 + 2 * b10 * f / 3 + f**2 / 5

    def sigma_v(self):
        return compute_sigma_v(self.k, self.P)

    def sigma0sq(self, z, b10=1.0, b01=0.0, sigma_p=0.0):
        D = self.growth_factor(z)
        f = self.growth_rate(z)
        sigmav = self.sigma_v()
        sv = sigmav * D

        sigma0_sq = (
            -1
            / (210 * (b10**2 + 2 * b10 * f / 3 + f**2 / 5))
            * (
                140 * b01 * (3 * b10 + f)
                + 2
                * (
                    -35 * b10**2 * (3 + f * (2 + f))
                    - 14 * b10 * f * (5 + 3 * f * (2 + f))
                    - 3 * f**2 * (7 + 5 * f * (2 + f))
                )
                * sv**2
                - (35 * b10**2 + 42 * b10 * f + 15 * f**2) * sigma_p**2
            )
        )
        return sigma0_sq

    def sigmaP2(self, k, z, ng, b10=1.0, rsd=True):
        """
        Parameters
        ----------
        k (float): Wavenumber at which to evaluate.
        z (float): Redshift.
        b10 (float, optional): Linear bias parameter (default 1.0).
        ng (float, optional): Number density of galaxies (default 1.0).

        Returns
        -------
        float
            Value of SigmaP2 at (k, z).
        """
        D = self.growth_factor(z)
        beta = 0.0
        if rsd:
            beta = self.growth_rate(z) / b10
        # Interpolate P(k) if needed
        P_interp = interp1d(
            self.k, self.P, kind="cubic", bounds_error=False, fill_value=0.0
        )

        mu_grid = np.linspace(-1, 1, 200)

        k_arr = np.atleast_1d(k)
        result = []
        for ki in k_arr:
            Pk = P_interp(ki)
            factor = D**2 * b10**2 * Pk * (1 + beta * mu_grid**2) ** 2
            integrand = 2 * (factor**2 + 2 * factor / ng)
            integral = np.trapezoid(integrand, mu_grid)
            result.append(0.5 * integral)
        return np.array(result)

    def cov_int(self, riN, rjN, z, delta_rN, ng, b10=1.0, rsd=True, iterpolator=None):
        # 1. Create k-array grid
        k_grid = np.logspace(-3, 2, 1000)

        # 2. Get sigmaP2 for all k
        if iterpolator is not None:
            sigmaP2_vals = iterpolator(k_grid)
        else:
            sigmaP2_vals = self.sigmaP2(k_grid, z, ng, b10, rsd)

        # 3. Compute j0_bar for all k
        j0_ri = j0_bar(k_grid, riN, delta_rN)
        j0_rj = j0_bar(k_grid, rjN, delta_rN)

        # 4. Compute integrand for all k
        integrand = k_grid**2 * sigmaP2_vals * j0_ri * j0_rj / (2 * np.pi) ** 3

        # 5. Integrate over k
        # result = np.trapezoid(integrand, k_grid)
        result = integrate.simpson(integrand, k_grid)
        return 4 * np.pi * result

    def cov_int_2d_vec(
        self, riN, rjN, z, delta_rN, ng, b10=1.0, rsd=True, iterpolator=None
    ):
        """
        Fully vectorized version of cov_int: riN and rjN can be 2D arrays (e.g., from meshgrid).
        """
        # k_grid is 1D
        k_grid = np.logspace(-3, 2, 1000)

        # sigmaP2_vals is 1D (shape: (Nk,))
        if iterpolator is not None:
            sigmaP2_vals = iterpolator(k_grid)
        else:
            sigmaP2_vals = self.sigmaP2(k_grid, z, ng, b10, rsd)

        # j0_bar returns (Nk, Nri, Nrj) if riN and rjN are 2D
        j0_ri = j0_bar(k_grid, riN, delta_rN)  # shape: (Nk, Nri, Nrj) or (Nk, Nri)
        j0_rj = j0_bar(k_grid, rjN, delta_rN)  # shape: (Nk, Nri, Nrj) or (Nk, Nrj)

        # If riN and rjN are 2D, j0_ri and j0_rj are (Nk, N, N)
        # Broadcast sigmaP2_vals to (Nk, 1, 1)
        sigmaP2_vals = sigmaP2_vals[:, None, None]

        integrand = (
            k_grid[:, None, None] ** 2 * sigmaP2_vals * j0_ri * j0_rj / (2 * np.pi) ** 3
        )

        # Integrate over k axis (axis=0)
        result = integrate.simpson(integrand, k_grid, axis=0)
        return 4 * np.pi * result


def bacco_params(cosmo_dict, expfactor=1):
    """
    Returns a dictionary of cosmological parameters for BACCO.
    omega_cold is the total cold matter (Baryons + Cold Dark Matter).
    """
    bacco_dict = {
        "omega_cold": cosmo_dict["Omega_m"],
        "sigma8_cold": cosmo_dict["sigma8"],
        "omega_baryon": cosmo_dict["Omega_b"],
        "ns": cosmo_dict["n_s"],
        "hubble": cosmo_dict["h"],
        "neutrino_mass": cosmo_dict.get("m_nu", 0),
        "w0": cosmo_dict.get("w0", -1.0),
        "wa": cosmo_dict.get("wa", 0.0),
        "expfactor": expfactor,
    }

    return bacco_dict


def _growth_factor_impl(z, Om):
    """
    Compute the linear growth factor D(z) for a given redshift `z` and matter density parameter `Om`.
    The growth factor is normalized at redshift z = 0.

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the growth factor.
    Om : float
        Matter density parameter (Ω_m).

    Returns
    -------
    D : float or ndarray
        The linear growth factor evaluated at the input redshift(s).
    """
    z = np.asarray(z)
    a = 1 / (1 + z)
    return (
        a
        * hyp2f1(1 / 3, 1, 11 / 6, (Om - 1) * a**3 / Om)
        / hyp2f1(1 / 3, 1, 11 / 6, (Om - 1) / Om)
    )


def _d_dz_growth_factor_impl(z_input, Om):
    """
    Computes the derivative of the linear growth factor D(z) with respect to redshift z.
    Parameters
    ----------
    z_input : float or array-like
              Redshift(s) at which to evaluate the derivative of the growth factor.
    Om : float
        Matter density parameter (Ω_m).

    Returns
    -------
    dDz_dz : float or ndarray
             The derivative of the linear growth factor D(z) with respect to z, evaluated at the input redshift(s).
    """

    z = np.asarray(z_input)
    a = 1 / (1 + z)

    dDz_dz = -hyp2f1(1.0 / 3, 1.0, 11.0 / 6, (Om - 1) * a**3 / Om) * a**2 / (
        hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, (Om - 1) / Om)
    ) - 6.0 / 11 * a**5 * (Om - 1) / Om * hyp2f1(
        4.0 / 3.0, 2.0, 17.0 / 6.0, (Om - 1) * a**3 / Om
    ) / hyp2f1(1.0 / 3.0, 1.0, 11.0 / 6.0, (Om - 1) / Om)

    return dDz_dz


def _growth_rate_impl(z_input, Om):
    """
    Computes the linear growth rate of cosmic structures as a function of redshift.

    Parameters
    ----------
    z_input : array-like or float
        Redshift(s) at which to evaluate the growth rate.
    Om : float
        Matter density parameter (Omega_m).

    Returns
    -------
    growth_rate : ndarray or float
        The linear growth rate evaluated at the input redshift(s).

    Notes
    -----
    The growth rate is defined as f(z) = dlnD/dlna, where D(z) is the linear growth factor.
    This implementation uses the derivative of the growth factor with respect to redshift.
    """
    z = np.asarray(z_input)
    a = 1 / (1 + z)
    Dz = _growth_factor_impl(z, Om)
    d_dz_Dz = _d_dz_growth_factor_impl(z, Om)
    return -d_dz_Dz / (a * Dz)


def growth_factor(z, Om):
    """
    Wrapper for _growth_factor_impl. See _growth_factor_impl docstring for details.
    """
    return _growth_factor_impl(z, Om)


def d_dz_growth_factor(z_input, Om):
    """
    Wrapper for _d_dz_growth_factor_impl. See _d_dz_growth_factor_impl docstring for details.
    """
    return _d_dz_growth_factor_impl(z_input, Om)


def growth_rate(z_input, Om):
    """
    Wrapper for _growth_rate_impl. See _growth_rate_impl docstring for details.
    """
    return _growth_rate_impl(z_input, Om)


def change_sigma8(k, P, sigma8_wanted):
    """
    Rescales the input power spectrum `P` so that the resulting spectrum yields the desired value of sigma8.

    Parameters
    ----------
    k : array_like
        Array of wavenumbers at which the power spectrum `P` is defined.
    P : array_like
        Power spectrum values corresponding to the wavenumbers `k`.
    sigma8_wanted : float
        The target value for sigma8, the RMS fluctuation of matter in spheres of radius 8 Mpc/h.

    Returns
    -------
    new_P : array_like
        The rescaled power spectrum such that its sigma8 matches `sigma8_wanted`.

    Raises
    ------
    ValueError
        If the computed sigma8 from the rescaled power spectrum does not match `sigma8_wanted` within a relative tolerance of 1e-3.
    """
    sigma8_old = compute_sigma8(k, P)
    new_P = P * (sigma8_wanted / sigma8_old) ** 2
    sigma8_computed = compute_sigma8(k, new_P)

    if not np.isclose(sigma8_computed, sigma8_wanted, rtol=1e-5):
        raise ValueError(
            f"sigma8_computed ({sigma8_computed}) does not match sigma8_wanted ({sigma8_wanted})"
        )

    return new_P


def compute_sigma8(k, P_arr):
    P_interp = interp1d(k, P_arr, kind="cubic", bounds_error=False, fill_value=0.0)

    def integrand(k_):
        return P_interp(k_) / (2.0 * np.pi) ** 3 * top_hat_filter(k_, 8) ** 2 * k_**2

    integral, _ = quad(integrand, 0.000001, 5, epsabs=0, epsrel=1e-8, limit=200)
    return np.sqrt(4.0 * np.pi * integral)


def compute_sigma_v(k, P):
    P_interp = interp1d(k, P, kind="cubic", bounds_error=False, fill_value=0.0)

    def integrand(k):
        return P_interp(k) / (2.0 * np.pi) ** 3

    integral, _ = quad(integrand, 0.001, 5, epsabs=0, epsrel=1e-8, limit=200)
    return np.sqrt(4.0 * np.pi / 3.0 * integral)


def Pk2xi(k, Pk, s_arr, z=0, Om=0.3):
    def integrand(x, Px, r, xpiv=1):
        return x**2 * Px / (2.0 * np.pi) ** 3 * j0(x * r) * wgc(x, xpiv, 4)

    s_arr = np.asarray(s_arr)
    xi = []
    for si in s_arr:
        xi.append(integrate.simpson(integrand(k, Pk, si), k))
    xi = 4 * np.pi * growth_factor(z, Om) ** 2 * np.array(xi)
    return xi


def Pk2xi_mcfit(k, Pk, s_arr):
    s_mc, xi_mc_result = mcfit.P2xi(k, lowring=True)(Pk, extrap=True)
    xi_intp = interp1d(s_mc, xi_mc_result, kind="cubic")
    xi = xi_intp(s_arr)
    return xi


def Pk2d_xi_ds(k, Pk, s_arr, z=0, Om=0.3):
    """
    Computes the derivative of the linear two-point correlation function xi(s) with respect to s,
    given a power spectrum P(k) and an array of separations s.

    Parameters
    ----------
    k : array_like
        Wavenumbers at which the power spectrum P(k) is defined.
    Pk : array_like
        Power spectrum values corresponding to the wavenumbers k.
    s_arr : array_like
        Array of separations at which to compute the derivative of xi.
    z : float, optional
        Redshift at which to evaluate the growth factor (default is 0).
    Om : float, optional
        Matter density parameter (default is 0.3).

    Returns
    -------
    d_xi_ds : ndarray
        The derivative of the two-point correlation function xi(s) with respect to s.
    """

    s_arr = np.asarray(s_arr)

    def integrand(x, Px_interp, r):
        return (
            x**2
            * Px_interp(x)
            / (2.0 * np.pi) ** 3
            * d_dr_j0_of_kr(x, r)
            * wgc(x, 1, 4)
        )

    Px_interp = interp1d(k, Pk, kind="cubic", bounds_error=False, fill_value=0.0)
    dxi = []
    for si in s_arr:
        val, _ = quad(lambda x: integrand(x, Px_interp, si), 0.001, 100, limit=200)
        dxi.append(val)
    dxi = 4 * np.pi * growth_factor(z, Om) ** 2 * np.array(dxi)

    return dxi


def dxi_ds(k, Pk, s_arr, z=0, Om=0.3):
    return np.gradient(Pk2xi(k, Pk, s_arr, z, Om), s_arr, edge_order=2)


def coefficient_form(k, Asq, sigma0sq):
    return Asq * np.exp(-(k**2) * sigma0sq)


def Asq(z, Om=0.3, b10=1.0):
    f = growth_rate(z, Om)
    return b10**2 + 2 * b10 * f / 3 + f**2 / 5


def sigma0sq(z, Om=0.3, b10=1.0, b01=0.0, sigmav=6.0, sigma_p=0.0):
    D = growth_factor(z, Om)
    f = growth_rate(z, Om)
    sv = sigmav * D

    sigma0_sq = (
        -1
        / (210 * (b10**2 + 2 * b10 * f / 3 + f**2 / 5))
        * (
            140 * b01 * (3 * b10 + f)
            + 2
            * (
                -35 * b10**2 * (3 + f * (2 + f))
                - 14 * b10 * f * (5 + 3 * f * (2 + f))
                - 3 * f**2 * (7 + 5 * f * (2 + f))
            )
            * sv**2
            - (35 * b10**2 + 42 * b10 * f + 15 * f**2) * sigma_p**2
        )
    )
    return sigma0_sq


def xiLS(N, Nr, dd_of_s, dr_of_s, rr_of_s):
    """
    Calculates the Landy-Szalay estimator for the two-point correlation function.

    Parameters
    ----------
    N : int
        Number of data points in the sample.
    Nr : int
        Number of random points in the sample.
    dd_of_s : array-like or float
        Number of data-data pairs as a function of separation s.
    dr_of_s : array-like or float
        Number of data-random pairs as a function of separation s.
    rr_of_s : array-like or float
        Number of random-random pairs as a function of separation s.

    Returns
    -------
    float or array-like
        The Landy-Szalay estimator value(s) for the two-point correlation function.
    """
    dd = dd_of_s / (N * (N - 1))
    dr = dr_of_s / (N * Nr)
    rr = rr_of_s / (Nr * (Nr - 1))
    return (dd - 2 * dr) / rr + 1


def xi_natural(N, Nr, dd_of_s, rr_of_s):
    """
    Calculates the natural estimator for the two-point correlation function.

    Parameters
    ----------
    N : int
        Number of data points in the sample.
    Nr : int
        Number of random points in the sample.
    dd_of_s : array-like or float
        Number of data-data pairs as a function of separation s.
    rr_of_s : array-like or float
        Number of random-random pairs as a function of separation s.

    Returns
    -------
    float or array-like
        The natural estimator value(s) for the two-point correlation function.
    """
    dd = dd_of_s / (N * (N - 1))
    rr = rr_of_s / (Nr * (Nr - 1))
    return dd / rr - 1


def Vs(ri, delta_r):
    """
    Computes the volume of a spherical shell with inner radius ri and thickness delta_r.
    Eq. (7) of Phys. Rev. D 99, 123515 (2019)
    """
    return np.pi / 3 * (12.0 * ri**2 * delta_r + delta_r**3)


def j0_bar(k, ri, delta_r):
    """
    Computes the average value of the spherical Bessel function j0 over a spherical shell of radius ri and thickness delta_r.
    Supports k (1D), ri (scalar, 1D, or 2D).
    """
    k = np.atleast_1d(k)
    ri = np.asarray(ri)
    r1 = ri - delta_r / 2.0
    r2 = ri + delta_r / 2.0

    # Broadcast k to match ri's shape
    # If ri is (N, N), k[:, None, None] will broadcast to (Nk, N, N)
    # If ri is (N,), k[:, None] will broadcast to (Nk, N)
    k_shape = (k.size,) + (1,) * ri.ndim
    k_b = k.reshape(k_shape)

    numerator = r2**2 * j1(k_b * r2) - r1**2 * j1(k_b * r1)
    denominator = k_b * Vs(ri, delta_r)
    result = 4.0 * np.pi * numerator / denominator
    return result


def cov_noPS_off_diag(ri, rj, delta_r):
    """
    Computes the covariance between two spherical shells with inner radii ri and rj, both with thickness delta_r.

    Parameters
    ----------
    ri : float
        The inner radius of the first shell.
    rj : float
        The inner radius of the second shell.
    delta_r : float
        The thickness of both shells.

    Returns
    -------
    float
        The covariance between the two shells.
    """

    cov_noPS_off_diag = (
        18
        / (np.pi * delta_r**2 * (12 * ri**2 + delta_r**2) * (12 * rj**2 + delta_r**2))
        * (
            2 * delta_r**2 * (ri + rj + abs(ri - rj))
            - 2
            * delta_r**2
            * ((ri - rj) ** 2 * (ri + rj) + abs(ri - rj) ** 3)
            / (ri - rj) ** 2
            - ri
            * delta_r
            * (2.0 * delta_r + abs(ri - rj + delta_r) - abs(-ri + rj + delta_r))
            - rj
            * delta_r
            * (2.0 * delta_r - abs(ri - rj + delta_r) + abs(-ri + rj + delta_r))
            + 2
            * ri
            * rj
            * (-2 * abs(ri - rj) + abs(ri - rj + delta_r) + abs(-ri + rj + delta_r))
            + rj
            * (
                -2 * (ri + rj) ** 2
                + (ri + rj - delta_r) ** 2
                + (ri + rj + delta_r) ** 2
                + 2 * (ri - rj) ** 2 * np.sign(ri - rj)
                - (-ri + rj + delta_r) ** 2 * np.sign(ri - rj - delta_r)
                - (ri - rj + delta_r) ** 2 * np.sign(ri - rj + delta_r)
            )
            + ri
            * (
                -2 * (ri + rj) ** 2
                + (ri + rj - delta_r) ** 2
                + (ri + rj + delta_r) ** 2
                - 2 * (ri - rj) ** 2 * np.sign(ri - rj)
                + (-ri + rj + delta_r) ** 2 * np.sign(ri - rj - delta_r)
                + (ri - rj + delta_r) ** 2 * np.sign(ri - rj + delta_r)
            )
        )
    )

    return cov_noPS_off_diag


def cov_noPS_diag(ri, delta_r):
    return 6.0 / (12.0 * np.pi * ri**2 + np.pi * delta_r**3)


def cov_noPS(ri, rj, delta_r):
    if ri == rj:
        return cov_noPS_diag(ri, delta_r)
    elif (
        ri != rj
        and delta_r**2 / (ri - rj) ** 2 <= 1.0
        and delta_r**2 / (ri + rj) ** 2 < 1.0
    ):
        return cov_noPS_off_diag(ri, rj, delta_r)


def cov_noPS_vec(ri, rj, delta_r):
    """
    Vectorized version of cov_noPS: ri and rj can be scalars or arrays of any shape.
    Applies cov_noPS_diag for diagonal, cov_noPS_off_diag for off-diagonal.
    """
    ri = np.asarray(ri)
    rj = np.asarray(rj)
    cov = np.zeros(np.broadcast(ri, rj).shape)

    # Diagonal mask
    diag_mask = np.isclose(ri, rj)
    # Off-diagonal mask (where analytic formula is valid)
    offdiag_mask = (~diag_mask) & (
        (delta_r**2 / (ri - rj) ** 2 <= 1.0) & (delta_r**2 / (ri + rj) ** 2 < 1.0)
    )

    # Apply diagonal formula
    cov[diag_mask] = cov_noPS_diag(ri[diag_mask], delta_r)
    # Apply off-diagonal formula
    cov[offdiag_mask] = cov_noPS_off_diag(ri[offdiag_mask], rj[offdiag_mask], delta_r)
    # All other cases remain zero (or you can set to np.nan if you prefer)
    return cov


if __name__ == "__main__":
    print(
        "This module provides cosmology utilities and is not intended to be run directly."
    )
