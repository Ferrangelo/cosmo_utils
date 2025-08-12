import numpy as np
from scipy import integrate, interpolate

from scipy.special import hyp2f1
from scipy.optimize import fsolve

from lp_utils.utils import SPEED_OF_LIGHT, read_json


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
            integral = integrate.quad(lambda x: 1.0 / self.E_correct(x), 0.0, z)[0]
        else:
            integral = np.array(
                [
                    integrate.quad(lambda x: 1.0 / self.E_correct(x), 0.0, z_i)[0]
                    for z_i in z
                ]
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
        integral = integrate.quad(lambda x: 1.0 / self.E_late_times(x), 0.0, z)
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
        distance_cubic_interp = interpolate.interp1d(z_vals, dist_vals, kind="cubic")
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

        volume_interp = interpolate.interp1d(z_vals, vol_grid, kind="cubic")
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

    def xi_natural(N, Nr, dd_of_s, dr_of_s, rr_of_s):
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

    def filt(q, R):
        return 3.0 * (np.sin(q * R) - q * R * np.cos(q * R)) / (q * R) ** 3

    integrand = P / (2.0 * np.pi) ** 3 * filt(k, 8) ** 2 * k**2
    sigma8_old = np.sqrt(4.0 * np.pi * integrate.simpson(integrand, k))

    new_P = P * (sigma8_wanted / sigma8_old) ** 2

    sigma8_computed = np.sqrt(
        4.0
        * np.pi
        * integrate.simpson(new_P / (2 * np.pi) ** 3 * filt(k, 8) ** 2 * k**2, k)
    )

    if not np.isclose(sigma8_computed, sigma8_wanted, rtol=1e-3):
        raise ValueError(
            f"sigma8_computed ({sigma8_computed}) does not match sigma8_wanted ({sigma8_wanted})"
        )

    return new_P


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
