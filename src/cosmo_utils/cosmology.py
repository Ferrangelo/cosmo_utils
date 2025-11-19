import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.integrate import quad
import mcfit

from scipy.special import hyp2f1
from scipy.optimize import fsolve

from cosmo_utils.utils import SPEED_OF_LIGHT, read_json, read_pk
from cosmo_utils.filters_et_functions import top_hat_filter, wgc, j0, d_dr_j0_of_kr
from cosmo_utils.filters_et_functions import j0_bar


class Cosmology:
    """
    A class for handling cosmological calculations and parameters.
    This class allows for the initialization of cosmological parameters either via a preset or by manual specification.
    It provides methods for computing cosmological quantities such as comoving distances, growth factors, and cosmological volumes.

    Parameters
    ----------
    preset : str, optional  Name of the preset cosmology to use. If provided, loads parameters from a corresponding JSON file.
    Omega_r : float, optional  Radiation density parameter.
    Omega_m : float, optional  Matter density parameter.
    Omega_DE : float, optional  Dark energy density parameter.
    Omega_k : float, optional  Curvature density parameter.
    h : float, optional  Dimensionless Hubble parameter.
    sigma8 : float, optional  RMS mass fluctuation amplitude at 8 Mpc/h.
    Pk_filename : str, optional  Filename for the power spectrum data file.

    Attributes
    ----------
    cosmo_id : str  Identifier for the chosen cosmology preset.
    Omega_r : float  Radiation density parameter.
    Omega_b : float  Baryon density parameter (if available in preset).
    Omega_m : float  Matter density parameter.
    Omega_DE : float  Dark energy density parameter.
    Omega_k : float  Curvature density parameter.
    h : float  Dimensionless Hubble parameter.
    sigma8 : float  RMS mass fluctuation amplitude at 8 Mpc/h.
    n_s : float  Scalar spectral index (if available in preset).
    w : float  Dark energy equation of state parameter w_0 in CPL parametrization (default -1.0).
    wa : float  Dark energy equation of state evolution parameter in CPL parametrization (default 0.0).
    As : float  Scalar amplitude (if available in preset).
    Pk_filename : str or None  Filename for the power spectrum (if provided).
    k : ndarray  Wavenumber array for the power spectrum (if Pk_filename is provided).
    Pk : ndarray  Power spectrum values from the file (if Pk_filename is provided).
    P : ndarray  Power spectrum values, possibly rescaled to match sigma8 (if Pk_filename is provided).

    Methods
    -------
    choose_cosmo(cosmology)  Loads cosmological parameters from a preset JSON file.
    set_pk()  Sets the power spectrum based on the provided Pk_filename.
    E_late_times(z)  Computes the dimensionless Hubble parameter E(z) for late times (ignoring radiation).
    E_correct(z)  Computes the dimensionless Hubble parameter E(z) including radiation.
    comoving_distance(z, h_units=True)  Calculates the comoving distance to redshift z, optionally in units of Mpc/h.
    comoving_distance_late_times(z, h_units=True)  Calculates the comoving distance to redshift z using late-time approximation.
    comoving_distance_interp(use_late_times=False, z_vals=None)  Returns an interpolator for comoving distance as a function of redshift.
    growth_factor(z_input, force_flat_lcdm_formula=False, force_use_colossus=False)  Computes the linear growth factor D(z) at given redshift(s).
    d_dz_growth_factor(z_input)  Computes the derivative of the linear growth factor D(z) with respect to redshift.
    growth_rate(z_input)  Computes the linear growth rate f(z) at given redshift(s).
    volume_zbin(zi, zf, fsky=None, solid_angle=None, use_late_times=False, z_vals=None)  Computes the comoving volume between two redshifts.
    get_vol_interp(zmin=0, zmax=2.5, fsky=None, solid_angle=None, z_vals=None)  Returns an interpolator for comoving volume vs redshift.
    find_z_for_target_volume(volume_target, fsky, z_min=0, z_max=2, z_vals=None)  Finds redshift for a target comoving volume.
    Pk2xi(s_arr, z=0)  Computes the linear two-point correlation function xi(s) from the power spectrum.
    Pk2xiNL(s_arr, z=0, rsd=False, *args, **kwargs)  Computes the nonlinear (Zeldovich approximation) two-point correlation function xi(s) from the power spectrum.
    binned_xiNL(s_arr, delta_r, z=0, rsd=False, *args, **kwargs)  Computes the binned nonlinear two-point correlation function.
    Pk2d_xi_ds(s_arr, z=0)  Computes the derivative of xi(s) with respect to s from P(k).
    dxi_ds(s_arr, z=0)  Computes the derivative of xi(s) with respect to s using numerical gradient.
    dxiNL_ds(s_arr, z=0, rsd=False, *args, **kwargs)  Computes the derivative of xiNL(s) with respect to s using numerical gradient.
    coefficient_form(z, rsd=False, *args, **kwargs)  Computes the nonlinear coefficient form for power spectrum transformation (Same name of Stefano's notebooks).
    Asq(z, rsd, b10=1.0)  Computes the amplitude squared for nonlinear corrections.
    sigma_v()  Computes the velocity dispersion sigma_v from the power spectrum.
    sigma0sq(z, rsd, b10=1.0, b01=0.0, sigma_p=0.0)  Computes sigma0 squared for nonlinear corrections.
    sigmaP2(k, z, ng, b10=1.0, rsd=False)  Computes SigmaP2(k,z) for covariance integrals.
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
            self.cosmo_id = preset
            self.cosmo = self.choose_cosmo(preset)
            self.h = self.cosmo["h"]
            self.Omega_r = self.cosmo["Omega_r"]
            self.Omega_b = self.cosmo.get("Omega_b")
            self.Omega_m = self.cosmo["Omega_m"]
            self.Omega_DE = self.cosmo["Omega_DE"]
            self.Omega_k = self.cosmo["Omega_k"]
            self.sigma8 = self.cosmo.get("sigma8")
            self.n_s = self.cosmo["n_s"]
            self.w = self.cosmo.get("w", -1.0)
            self.wa = self.cosmo.get("wa", 0.0)
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
            self.Omega_k = Omega_k
            self.sigma8 = sigma8
            self.Pk_filename = Pk_filename

        self.set_pk()

    def choose_cosmo(self, cosmology):
        """
        Loads cosmological parameters from a preset JSON file.

        Parameters
        ----------
        cosmology : str  Name of the preset cosmology.

        Returns
        -------
        cosmo_dict : dict  Dictionary of cosmological parameters.
        """
        list_of_cosmologies = [
            "raygal",
            "raygal_wcdm",
            "istf",
            "wmap1",
            "wmap3",
            "wmap5",
            "wmap7",
            "wmap9",
            "fs2",
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
        filename : str  Path to the JSON file.

        Returns
        -------
        cosmo_dict : dict  Dictionary of cosmological parameters.
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
            if self.sigma8 is None:
                self.sigma8 = sigma8_computed
                self.P = self.Pk
                print("\n--------- Set sigma8 from Pk ---------")
                print(f"sigma8 = {self.sigma8:.8f}\n")
            elif np.isclose(sigma8_computed, self.sigma8, rtol=1e-6):
                self.P = self.Pk
            else:
                print(
                    f"Warning: Computed sigma8 ({sigma8_computed}) does not match provided sigma8 ({self.sigma8}). "
                    "Rescaling the power spectrum to match the provided sigma8."
                )
                self.P = change_sigma8(self.k, self.Pk, self.sigma8)
                print("\n--------- After rescaling ---------")
                print(f"sigma8 = {self.sigma8:.8f}")
                print(f"sigma8 from Pk = {compute_sigma8(self.k, self.P):.8f}\n")

    def E_late_times(self, z):
        """
        Calculate the dimensionless Hubble parameter E(z) at late times.

        Parameters
        ----------
        z : float or array-like  Redshift(s) at which to evaluate E(z).

        Returns
        -------
        float or ndarray  E(z) = sqrt(Omega_m(1+z)^3 + Omega_k(1+z)^2 + Omega_DE * (1+z)^{3(1 + w0 + wa)} exp{-3 wa z / (1 + z)}).
        """
        # Use of the CPL parametrization
        w0 = self.w
        wa = self.wa

        return np.sqrt(
            self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_DE
            * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
        )

    def E_correct(self, z):
        """
        Calculate the dimensionless Hubble parameter E(z) including radiation.

        Parameters
        ----------
        z : float or array-like  Redshift value(s) at which to evaluate E(z).

        Returns
        -------
        float or ndarray  E(z) including radiation, matter, curvature, dark energy.

        Notes
        -----
        E(z) = sqrt(Omega_r(1+z)^4 + Omega_m(1+z)^3 + Omega_k(1+z)^2 + Omega_DE * (1+z)^{3(1 + w0 + wa)} exp{-3 wa z / (1 + z)}).
        """
        # Use of the CPL parametrization
        w0 = self.w
        wa = self.wa

        return np.sqrt(
            self.Omega_r * (1 + z) ** 4
            + self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_DE
            * (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
        )

    def comoving_distance(self, z, h_units=True):
        """
        Calculate the comoving distance to a given redshift z.

        Parameters
        ----------
        z : float or array-like  Redshift(s) for distance computation.
        h_units : bool, optional  If True return Mpc/h, else Mpc (default True).

        Returns
        -------
        float or ndarray  Comoving distance(s).
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
        z : float  Redshift.
        h_units : bool, optional  If True return Mpc/h (default True).

        Returns
        -------
        float  Comoving distance.
        """
        h = 1.0
        if not h_units:
            h = self.h
        Dh = SPEED_OF_LIGHT * 0.01 / h
        integral = quad(lambda x: 1.0 / self.E_late_times(x), 0.0, z)
        return Dh * integral[0]

    def comoving_distance_interp(self, use_late_times=False, z_vals=None):
        """
        Build cubic interpolation for comoving distance vs redshift.

        Parameters
        ----------
        use_late_times : bool, optional  If True use late-time approximation (default False).
        z_vals : array-like or None, optional  Redshift grid (default linspace 0..2.5, 4000 pts).

        Returns
        -------
        callable  Interpolator r(z).
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

    def growth_factor(self, z_input, force_flat_lcdm_formula=False, force_use_colossus=False):
        """
        Wrapper for _growth_factor_impl. See _growth_factor_impl docstring for details.
        """
        if force_flat_lcdm_formula:
            return _growth_factor_lcdm_impl(z_input, self.Omega_m)
        elif force_use_colossus:
            return self._growth_factor_impl(z_input)
        elif np.isclose(self.w, -1.0) and np.isclose(self.wa, 0.0) and np.isclose(self.Omega_k, 0.0):
            return _growth_factor_lcdm_impl(z_input, self.Omega_m)
        else:
            return self._growth_factor_impl(z_input)
   
    def _growth_factor_impl(self, z_eval):
        """
        Compute exact normalized growth factor D(z) using Colossus for the current cosmology.

        Notes:
        - Requires a baryon density Omega_b to build the Colossus cosmology.
        - Returns D(z) normalized to z=0 (same convention as Colossus.growthFactor).
        """
        try:
            from colossus.cosmology import cosmology as col_cosmo
        except ImportError as e:
            raise ImportError(
                "Colossus is required for exact growth factor. Install with `pip install colossus` "
                "or add it to PYTHONPATH."
            ) from e

        # Gather parameters for Colossus
        Om0 = self.Omega_m
        H0 = 100.0 * self.h
        ns = self.n_s
        sigma8 = self.sigma8
        w0 = self.w 
        wa = self.wa

        # Baryon density is required by Colossus
        Ob0 = getattr(self, "Omega_b", None)
        if Ob0 is None and hasattr(self, "cosmo") and isinstance(self.cosmo, dict):
            Ob0 = self.cosmo.get("Omega_b", None)
        if Ob0 is None:
            print("Omega_b0 (baryon density) is required for Colossus to instantiate a cosmology.")
            print("Making up a value of Omega_b0 = 0.05, as it does not affect the computation of the growth factor.")
            Ob0 = 0.05

        Ob0 = float(Ob0)

        flat = bool(abs(self.Omega_k) < 1e-12)
        Ode0 = None if flat else float(self.Omega_DE)

        # Choose Colossus DE model
        if np.isclose(w0, -1.0) and np.isclose(wa, 0.0):
            de_model = "lambda"
            w0_arg, wa_arg = None, None
        elif np.isclose(wa, 0.0):
            de_model = "w0"
            w0_arg, wa_arg = float(w0), None
        else:
            de_model = "w0wa"
            w0_arg, wa_arg = float(w0), float(wa)

        # Build or reuse a Colossus cosmology object (simple cache)
        key = (
            flat, Om0, Ode0, Ob0, H0, sigma8, ns,
            de_model, w0_arg, wa_arg
        )
        cache_ok = hasattr(self, "_colossus_cache") and self._colossus_cache.get("key") == key
        if not cache_ok:
            kwargs = dict(
                name="cosmo_utils_tmp",
                flat=flat, Om0=Om0, Ob0=Ob0, H0=H0,
                sigma8=sigma8, ns=ns,
                de_model=de_model,
                interpolation=True,
                # Avoid disk persistence to keep things local to the session
                persistence='',
                print_info=False, print_warnings=False,
            )
            if not flat:
                kwargs["Ode0"] = Ode0
            if w0_arg is not None:
                kwargs["w0"] = w0_arg
            if wa_arg is not None:
                kwargs["wa"] = wa_arg

            col = col_cosmo.Cosmology(**kwargs)
            self._colossus_cache = {"key": key, "obj": col}
        else:
            col = self._colossus_cache["obj"]

        return col.growthFactor(z_eval)


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
        Compute comoving volume between redshifts zi and zf over a sky area.

        Parameters
        ----------
        zi : float  Initial redshift.
        zf : float  Final redshift.
        fsky : float, optional  Sky fraction (0<f<=1); overrides solid_angle.
        solid_angle : float, optional  Solid angle in steradians if fsky None.
        use_late_times : bool, optional  Use late-time distance (default False).
        z_vals : array-like, optional  Redshift grid for interpolation.

        Returns
        -------
        float  Comoving volume between zi and zf.

        Raises
        ------
        ValueError  If neither fsky nor solid_angle provided.
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
        Create interpolator for cumulative comoving volume.

        Parameters
        ----------
        zmin : float, optional  Minimum redshift (default 0).
        zmax : float, optional  Maximum redshift (default 2.5).
        fsky : float, optional  Sky fraction; computed from solid_angle if None.
        solid_angle : float, optional  Solid angle steradians.
        z_vals : array-like, optional  Redshift grid (default 4000 pts 0..2.5).

        Returns
        -------
        scipy.interpolate.interp1d  Interpolator giving volume at z.
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
        Find redshift where cumulative volume equals target.

        Parameters
        ----------
        volume_target : float  Target comoving volume.
        fsky : float  Sky fraction (0<f<=1).
        z_min : float, optional  Minimum search redshift (default 0).
        z_max : float, optional  Maximum search redshift (default 2).
        z_vals : array-like, optional  Redshift grid for distances.

        Returns
        -------
        float  Redshift matching target volume.
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

    def Pk2xiNL(self, s_arr, z=0, rsd=False, *args, **kwargs):
        """
        Compute the nonlinear two-point correlation function xi(s) from a power spectrum P(k).

        Parameters
        ----------
        s_arr : array_like
            One-dimensional array of comoving separations s at which to evaluate xi(s).
        z : float, optional
            Redshift at which to evaluate the growth factor and any redshift-dependent
            coefficient_form. Default is 0.
        rsd : bool, optional
            Reserved parameter for redshift-space distortion handling.
        *args, **kwargs
            Extra positional and keyword arguments are forwarded to
            self.coefficient_form(z, rsd, *args, **kwargs) which then passes them to 
            sigma0sq(self, z, rsd, b10=1.0, b01=0.0, sigma_p=0.0). These extra parameters
            are b10, b01, sigma_p. b01 and sigma_p are only relevant if rsd=True.

        Returns
        -------
        xi : numpy.ndarray
            1D array of xi(s) values corresponding to the input s_arr. The computed array
            has the same shape as s_arr. The returned xi(s) equals
            4*pi * [D(z)]^2 * integral(...) where the integral is computed by
            scipy.integrate.simpson over the self.k grid.
        """
        def integrand(x, Px, r, xpiv=1):
            return (
                x**2
                * self.coefficient_form(z, rsd, *args, **kwargs)
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

    def binned_xiNL(self, s_arr, delta_r, z=0, rsd=False, *args, **kwargs):
        """
        Compute the binned non-linear two-point correlation function xi_NL(s).

        This method computes a binned version of the non-linear correlation function
        by integrating the (weighted) power spectrum against a bin-averaged
        spherical Bessel kernel for each separation value in `s_arr`. The
        integration over wavenumber k is performed with Simpson's rule.

        Parameters
        ----------
        s_arr : array_like
            One-dimensional array or scalar of separations (s) at which to evaluate
            the binned correlation function. The returned array has the same shape
            as `s_arr`. 
        delta_r : float
            Width of the radial bin used to compute the bin-averaged spherical
            Bessel function j0_bar(r, delta_r). Must be positive.
        z : float, optional
            Redshift at which to evaluate the correlation function. Defaults to 0.
        rsd : bool, optional
            Reserved parameter for redshift-space distortion handling.
        *args, **kwargs
            Extra positional and keyword arguments are forwarded to
            self.coefficient_form(z,rsd, *args, **kwargs) which then passes them to 
            sigma0sq(self, z, rsd,  b10=1.0, b01=0.0, sigma_p=0.0). These extra parameters
            are b10, b01, sigma_p. b01 and sigma_p are only relevant if rsd=True.

        Returns
        -------
        xi : numpy.ndarray
            Array of the same shape as `s_arr` containing the binned non-linear
            correlation function xi_NL(s) evaluated at each input separation.
        """
        def integrand(x, Px, r, xpiv=1):
            return (
                x**2
                * self.coefficient_form(z, rsd, *args, **kwargs)
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
        Derivative of xi(s) from P(k).

        Parameters
        ----------
        s_arr : array_like  Separations where derivative is computed.
        z : float, optional  Redshift (default 0).

        Returns
        -------
        ndarray  d xi / ds at s.
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

    def dxiNL_ds(self, s_arr, z=0, rsd=False, *args, **kwargs):
        return np.gradient(self.Pk2xiNL(s_arr, z, rsd, *args, **kwargs), s_arr, edge_order=2)

    def coefficient_form(self, z, rsd=False, *args, **kwargs):
        if len(args) >= 1:
            b10 = args[0]
        else:
            b10 = kwargs.get("b10", 1.0)

        sigma0sq = self.sigma0sq(z, rsd, *args, **kwargs)
        Asq = self.Asq(z, rsd, b10)

        return Asq * np.exp(-(self.k**2) * sigma0sq)

    def Asq(self, z, rsd, b10=1.0):
        f = self.growth_rate(z) if rsd else 0.0
        return b10**2 + 2 * b10 * f / 3 + f**2 / 5

    def sigma_v(self):
        return compute_sigma_v(self.k, self.P)

    def sigma0sq(self, z, rsd, b10=1.0, b01=0.0, sigma_p=0.0):
        D = self.growth_factor(z)
        f = self.growth_rate(z) if rsd else 0.0
        sigmav = self.sigma_v()
        sv = sigmav * D
        
        if not rsd:
            b01 = 0.0
            sigma_p = 0.0

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

    def sigmaP2(self, k, z, ng, b10=1.0, rsd=False):
        """
        Compute SigmaP2(k,z) for covariance integrals.

        Parameters
        ----------
        k : float or array_like  Wavenumber(s).
        z : float  Redshift.
        ng : float  Number density of galaxies.
        b10 : float, optional  Linear bias (default 1.0).
        rsd : bool, optional  Include RSD factor (default True).

        Returns
        -------
        ndarray  SigmaP2 values matching input k shape.
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


def bacco_params(cosmo_dict, expfactor=1):
    """
    Build BACCO parameter dictionary.

    Parameters
    ----------
    cosmo_dict : dict  Cosmological parameters.
    expfactor : float, optional  Scale factor a (default 1).

    Returns
    -------
    dict  BACCO-formatted parameter dictionary.
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

def _growth_factor_lcdm_impl(z, Om):
    """
    Linear growth factor D(z), normalized at z=0.

    Parameters
    ----------
    z : float or array-like  Redshift(s).
    Om : float  Matter density parameter.

    Returns
    -------
    float or ndarray  Growth factor D(z).
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
    Derivative dD/dz of the linear growth factor.

    Parameters
    ----------
    z_input : float or array-like  Redshift(s).
    Om : float  Matter density parameter.

    Returns
    -------
    float or ndarray  dD/dz at input redshift(s).
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
    Linear growth rate f(z) = d ln D / d ln a.

    Parameters
    ----------
    z_input : float or array-like  Redshift(s).
    Om : float  Matter density parameter.

    Returns
    -------
    float or ndarray  Growth rate f(z).
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
    Rescale P(k) to achieve desired sigma8.

    Parameters
    ----------
    k : array_like  Wavenumbers.
    P : array_like  Power spectrum values.
    sigma8_wanted : float  Target sigma8.

    Returns
    -------
    array_like  Rescaled power spectrum.

    Raises
    ------
    ValueError  If resulting sigma8 mismatch beyond tolerance.
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
    """
    Compute sigma8 from P(k).
    Parameters
    ----------
    k : array_like  Wavenumbers in h/Mpc.
    P_arr : array_like  Power spectrum values in (Mpc/h)^3.
    """
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
    Derivative d xi / ds from P(k).

    Parameters
    ----------
    k : array_like  Wavenumbers.
    Pk : array_like  Power spectrum values.
    s_arr : array_like  Separations.
    z : float, optional  Redshift (default 0).
    Om : float, optional  Matter density parameter (default 0.3).

    Returns
    -------
    ndarray  d xi / ds at s.
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


def Vs(ri, delta_r):
    """
    Computes the volume of a spherical shell with inner radius ri and thickness delta_r.
    Eq. (7) of Phys. Rev. D 99, 123515 (2019)
    """
    return np.pi / 3 * (12.0 * ri**2 * delta_r + delta_r**3)


def xiLS(N, Nr, dd_of_s, dr_of_s, rr_of_s):
    """
    Landy-Szalay estimator for xi.

    Parameters
    ----------
    N : int  Number of data points.
    Nr : int  Number of random points.
    dd_of_s : array-like or float  Data-data pair counts.
    dr_of_s : array-like or float  Data-random pair counts.
    rr_of_s : array-like or float  Random-random pair counts.

    Returns
    -------
    float or array-like  Landy-Szalay xi(s).
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


if __name__ == "__main__":
    print(
        "This module provides cosmology utilities and is not intended to be run directly."
    )
