"""Linear Point (LP) analysis utilities.

This module provides tools to:

* Generate multivariate Gaussian mock correlation-function (xi) realizations.
* Compute analytical / semi-analytical covariance elements for binned statistics.
* Perform generalized least squares (GLS) polynomial fits to xi(s).
* Locate the BAO dip, peak, and derive the Linear Point (LP) scale.
* Propagate polynomial coefficient uncertainties to dip / peak / LP errors analytically.
* Aggregate LP statistics across many realizations (bias, scatter, KS tests, etc.).

Key classes / functions exported (see ``__all__``):
    - ``create_random_xi_data`` & ``RandomXiResult`` for mock generation.
    - ``PolynomialModel`` & ``gls_polynomial_fit`` for GLS fitting with arbitrary
        active coefficient masks.
    - ``find_dip_peak`` & ``linear_point`` to extract extrema and LP scale.
    - ``propagate_lp_uncertainty`` for analytic error propagation.
    - ``process_realization`` and ``aggregate_lp_statistics`` to run the full
        pipeline over single / multiple realizations.

All polynomial coefficients follow numpy.polynomial's ascending-order convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union, Sequence

import warnings
import numpy as np
import numpy.linalg as npl
from scipy import integrate, stats
from lp_utils.filters_et_functions import j1

ArrayLike = Union[np.ndarray, Iterable[float]]

__all__ = [
    "create_random_xi_data",
    "RandomXiResult",
    "PolynomialModel",
    "gls_polynomial_fit",
    "gls_polynomial_fit_cholesky",
    "find_dip_peak",
    "linear_point",
    "propagate_lp_uncertainty",
    "process_realization",
    "aggregate_lp_statistics",
]


@dataclass
class RandomXiResult:
    """Container for generated mock catalogs.

    Attributes
    ----------
    samples : ndarray, shape (n_samples, n_points)
        The simulated realizations (each row is one mock).
    mean : ndarray, shape (n_points,)
        Mean vector used (xi values).
    covariance : ndarray, shape (n_points, n_points)
        Covariance matrix actually employed.
    seed : Optional[int]
        RNG seed used (if provided by user).
    """

    samples: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    seed: Optional[int]


def _extract_mean_vector(xi: ArrayLike) -> np.ndarray:
    """Extract the mean xi vector from input.

    Accepts either:
      * 1-D array-like of length N (already the mean xi vector), or
      * 2-D array of shape (N,2) where second column is xi(s).
    """
    arr = np.asarray(xi)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1].astype(float)
    raise ValueError(
        "xi must be 1-D (N,) mean vector or 2-D (N,2) array whose 2nd column is xi(s)."
    )


def create_random_xi_data(
    xi: ArrayLike,
    covariance: np.ndarray,
    n_samples: int = 300,
    seed: Optional[int] = None,
    check_consistency: bool = True,
) -> RandomXiResult:
    """Generate multivariate Gaussian mock ``xi`` realizations.

    Parameters
    ----------
    xi : array-like
        Either an (N,) mean vector, or an (N,2) array where the second column
        holds xi(s) values (first column typically the scale ``s``).
    covariance : ndarray (N,N)
        Covariance matrix corresponding to the mean vector (same ordering).
    n_samples : int, optional
        Number of mock realizations to generate (default 300).
    seed : int, optional
        Seed for reproducibility. If ``None`` a random seed is used.
    check_consistency : bool, optional
        If True (default) verify that diagonal of covariance is non-negative and
        shapes match; raises on mismatch.

    Returns
    -------
    RandomXiResult
        Dataclass bundling samples, mean vector, covariance, and diagnostics.

    Raises
    ------
    ValueError
        If input shapes mismatch or covariance invalid.
    """
    mean_vec = _extract_mean_vector(xi)
    cov = np.asarray(covariance, dtype=float)

    if check_consistency:
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance must be square")
        if cov.shape[0] != mean_vec.size:
            raise ValueError(
                f"Covariance dimension {cov.shape[0]} does not match length (binning) of the 2pcf {mean_vec.size}."
            )
        if np.any(np.diag(cov) < 0):
            raise ValueError("Covariance has negative variances on its diagonal.")

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean_vec, cov, size=n_samples, check_valid="warn", tol=1e-8
    )

    return RandomXiResult(
        samples=samples,
        mean=mean_vec,
        covariance=cov,
        seed=seed,
    )


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
    Covariance between two distinct spherical shells.

    Parameters
    ----------
    ri : float  Inner radius first shell.
    rj : float  Inner radius second shell.
    delta_r : float  Shell thickness.

    Returns
    -------
    float  Off-diagonal covariance value.
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
    # Broadcast inputs to a common shape to safely apply boolean masks
    Ri, Rj = np.broadcast_arrays(ri, rj)
    cov = np.zeros_like(Ri, dtype=float)

    # Diagonal mask
    diag_mask = np.isclose(Ri, Rj)
    # Off-diagonal mask (where analytic formula is valid)
    offdiag_mask = (~diag_mask) & (
        (delta_r**2 <= (Ri - Rj) ** 2) & (delta_r**2 < (Ri + Rj) ** 2)
    )

    # Apply diagonal formula
    cov[diag_mask] = cov_noPS_diag(Ri[diag_mask], delta_r)
    # Apply off-diagonal formula
    cov[offdiag_mask] = cov_noPS_off_diag(Ri[offdiag_mask], Rj[offdiag_mask], delta_r)
    # All other cases remain zero (or you can set to np.nan if you prefer)
    return cov


def cov_int(cosmo, riN, rjN, z, delta_rN, ng, b10=1.0, rsd=True, iterpolator=None):
    # 1. Create k-array grid
    k_grid = np.logspace(-3, 2, 1000)

    # 2. Get sigmaP2 for all k
    if iterpolator is not None:
        sigmaP2_vals = iterpolator(k_grid)
    else:
        sigmaP2_vals = cosmo.sigmaP2(k_grid, z, ng, b10, rsd)

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
    cosmo, riN, rjN, z, delta_rN, ng, b10=1.0, rsd=True, iterpolator=None
):
    """
    Fully vectorized covariance integral for 2D grids.

    Parameters
    ----------
    riN : array_like  Radial bin centers (can be 2D).
    rjN : array_like  Radial bin centers (can be 2D).
    z : float  Redshift.
    delta_rN : float  Bin thickness.
    ng : float  Number density.
    b10 : float, optional  Linear bias (default 1.0).
    rsd : bool, optional  Include RSD (default True).
    iterpolator : callable, optional  Precomputed SigmaP2(k) interpolator.

    Returns
    -------
    ndarray  Covariance matrix/grid.
    """
    # k_grid is 1D
    k_grid = np.logspace(-3, 2, 1000)

    # sigmaP2_vals is 1D (shape: (Nk,))
    if iterpolator is not None:
        sigmaP2_vals = iterpolator(k_grid)
    else:
        sigmaP2_vals = cosmo.sigmaP2(k_grid, z, ng, b10, rsd)

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


# ---------------------------------------------------------------------------
# Polynomial GLS fitting & Linear Point extraction utilities
# ---------------------------------------------------------------------------


@dataclass
class PolynomialModel:
    """Simple polynomial model f(s) = Σ_{k=0}^order a_k s^k.

    Parameters
    ----------
    order : int
        Maximum polynomial degree.
    active_mask : Optional[Sequence[bool]]
        Mask of length order+1 indicating which coefficients are *fit*.
        If None, all coefficients are active.
    """

    order: int
    active_mask: Optional[Sequence[bool]] = None

    def __post_init__(self):
        if self.order < 0:
            raise ValueError("order must be >= 0")
        if self.active_mask is None:
            self.active_mask = [True] * (self.order + 1)
        if len(self.active_mask) != self.order + 1:
            raise ValueError("active_mask length mismatch")

    @property
    def n_params(self) -> int:
        return sum(self.active_mask)

    def design_matrix(self, s: np.ndarray) -> np.ndarray:
        """Return design matrix X of shape (N, n_active)."""
        s = np.asarray(s)
        # Columns in ascending power order
        full = np.vstack([s**k for k in range(self.order + 1)]).T  # (N, order+1)
        mask = np.array(self.active_mask, dtype=bool)
        return full[:, mask]

    def expand_params(self, active_params: np.ndarray) -> np.ndarray:
        """Embed active parameter vector into full coefficient array."""
        coeffs = np.zeros(self.order + 1, dtype=float)
        mask = np.array(self.active_mask, dtype=bool)
        coeffs[mask] = active_params
        return coeffs

    def evaluate(self, s: np.ndarray, params: np.ndarray) -> np.ndarray:
        coeffs = self.expand_params(params)
        # Use Horner / vectorized evaluation
        # numpy.polynomial.polynomial.polyval expects ascending order
        from numpy.polynomial import polynomial as P

        return P.polyval(np.asarray(s), coeffs)

    def derivative_coeffs(self, coeffs: np.ndarray, order: int = 1) -> np.ndarray:
        """Return coefficients of the ``order``-th derivative of a polynomial.

        Parameters
        ----------
        coeffs : array_like
            Polynomial coefficients in ascending order (c0, c1, c2, ...).
        order : int, optional
            Non-negative derivative order to take (default 1). ``order=0`` returns
            a copy of the original coefficient array.

        Returns
        -------
        ndarray
            Coefficients (ascending order) of the requested derivative. Length is
            ``max(len(coeffs) - order, 1)`` because each derivative lowers the
            degree by one until reaching a constant.

        Raises
        ------
        ValueError
            If ``order`` is negative.
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        from numpy.polynomial import polynomial as P

        dc = np.array(coeffs, copy=True)
        for _ in range(order):
            dc = P.polyder(dc)
        return dc

    def real_derivative_roots(
        self, coeffs: np.ndarray, deriv_order: int = 1
    ) -> np.ndarray:
        from numpy.polynomial import polynomial as P

        dcoeffs = self.derivative_coeffs(coeffs, deriv_order)
        roots = P.polyroots(dcoeffs)
        roots_real = roots[np.isclose(roots.imag, 0.0)].real
        return np.sort(roots_real)


@dataclass
class GLSFitResult:
    params: np.ndarray  # active params (coefficient of the polynomial)
    full_coeffs: np.ndarray  # full coefficient array length order+1
    param_cov: np.ndarray  # covariance of active params
    chi2: float
    dof: int
    success: bool
    message: str = ""


def gls_polynomial_fit(
    s: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
    model: PolynomialModel,
) -> GLSFitResult:
    """Generalized least squares fit of polynomial to data with covariance.

    Equivalent to solving (X^T C^{-1} X) a = X^T C^{-1} y for active params.
    """
    s = np.asarray(s)
    y = np.asarray(y)
    C = np.asarray(covariance)
    if C.shape[0] != C.shape[1] or C.shape[0] != s.size:
        raise ValueError("Covariance shape mismatch")
    try:
        Ci = npl.inv(C)
    except npl.LinAlgError:
        # Regularize minimally
        jitter = 1e-12 * np.trace(C) / C.shape[0]
        Ci = npl.inv(C + jitter * np.eye(C.shape[0]))

    X = model.design_matrix(s)
    XT_Ci = X.T @ Ci
    alpha = XT_Ci @ X
    beta = XT_Ci @ y
    try:
        params = npl.solve(alpha, beta)
        param_cov = npl.inv(alpha)
        y_model = X @ params
        resid = y - y_model
        chi2 = float(resid.T @ Ci @ resid)
        dof = y.size - model.n_params
        return GLSFitResult(
            params, model.expand_params(params), param_cov, chi2, dof, True
        )
    except npl.LinAlgError as e:
        return GLSFitResult(
            np.full(model.n_params, np.nan),
            np.full(model.order + 1, np.nan),
            np.full((model.n_params, model.n_params), np.nan),
            np.nan,
            0,
            False,
            f"Solve failed: {e}",
        )


def gls_polynomial_fit_cholesky(
    s: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
    model: PolynomialModel,
) -> GLSFitResult:
    """Generalized least squares fit using Cholesky *whitening* for stability.

    This is numerically equivalent to :func:`gls_polynomial_fit`, but instead of
    forming an explicit inverse of the covariance matrix it performs a Cholesky
    factorization ``C = L L^T`` and solves an *ordinary* least squares problem in
    the whitened space:

        minimize || L^{-1} (X a - y) ||^2

    The active parameter handling (masking) is identical to the standard GLS
    routine, and the returned ``GLSFitResult`` fields have the same meaning.

    Parameters
    ----------
    s : array_like
        1-D array of abscissa (e.g. separation) values.
    y : array_like
        1-D array of observed function values to fit.
    covariance : (N,N) array_like
        Data covariance matrix aligned with ``s`` / ``y`` ordering.
    model : PolynomialModel
        Polynomial model instance specifying order and active coefficients.

    Returns
    -------
    GLSFitResult
        Fit result container: parameters, covariance, chi2, etc.

    Notes
    -----
    * The parameter covariance is ``(X^T C^{-1} X)^{-1} = (X_w^T X_w)^{-1}`` where
      ``X_w = L^{-1} X``.
    * If the Cholesky factorization fails (matrix not PD), a tiny diagonal jitter
      is added adaptively until success or a maximum number of attempts is reached.
    """
    s = np.asarray(s)
    y = np.asarray(y)
    C = np.asarray(covariance)
    if C.shape[0] != C.shape[1] or C.shape[0] != s.size:
        raise ValueError("Covariance shape mismatch")

    # Build design matrix of *active* parameters
    X = model.design_matrix(s)

    # Attempt Cholesky with mild adaptive jitter if necessary
    jitter_base = 1e-14 * np.trace(C) / C.shape[0]
    L = None
    attempts = 0
    while attempts < 5:
        try:
            L = npl.cholesky(C + (jitter_base * (10**attempts)) * np.eye(C.shape[0]))
            break
        except npl.LinAlgError:
            attempts += 1
    if L is None:
        return GLSFitResult(
            np.full(model.n_params, np.nan),
            np.full(model.order + 1, np.nan),
            np.full((model.n_params, model.n_params), np.nan),
            np.nan,
            0,
            False,
            "Cholesky factorization failed",
        )

    # Whiten: solve L z = v  (forward substitution) => z = L^{-1} v
    # Use solve for each column / vector.
    from scipy.linalg import solve_triangular  # local import (lightweight)

    y_w = solve_triangular(L, y, lower=True, overwrite_b=False, check_finite=False)
    # For X, whiten each column
    X_w = solve_triangular(L, X, lower=True, overwrite_b=False, check_finite=False)

    # Solve ordinary least squares in whitened space using QR for stability
    try:
        Q, R = np.linalg.qr(X_w, mode="reduced")
        params = npl.solve(R, Q.T @ y_w)
        # Param covariance: (X_w^T X_w)^{-1}; compute via R since X_w = Q R
        # X_w^T X_w = R^T R  ==> (X_w^T X_w)^{-1} = R^{-1} (R^{-1})^T
        Rinv = npl.inv(R)
        param_cov = Rinv @ Rinv.T
        resid_w = y_w - X_w @ params
        chi2 = float(resid_w @ resid_w)
        dof = y.size - model.n_params
        return GLSFitResult(
            params,
            model.expand_params(params),
            param_cov,
            chi2,
            dof,
            True,
            "",
        )
    except npl.LinAlgError as e:
        return GLSFitResult(
            np.full(model.n_params, np.nan),
            np.full(model.order + 1, np.nan),
            np.full((model.n_params, model.n_params), np.nan),
            np.nan,
            0,
            False,
            f"QR solve failed: {e}",
        )


@dataclass
class ExtremumResult:
    dip: float
    peak: float
    lp: float
    success: bool
    message: str = ""


def find_dip_peak(
    coeffs: np.ndarray,
    s_min: float,
    s_max: float,
    dip_window: Optional[tuple] = None,
    peak_window: Optional[tuple] = None,
    tol_second: float = 0.0,
) -> ExtremumResult:
    """Locate dip (min) and peak (max) of polynomial inside interval by roots of derivative.

    Windows (a,b) restrict acceptable root locations. Second derivative sign classifies extrema.
    """
    from numpy.polynomial import polynomial as P

    dcoeffs = P.polyder(coeffs)
    roots = P.polyroots(dcoeffs)
    real_roots = roots[np.isclose(roots.imag, 0)].real
    # Filter within main interval
    in_range = real_roots[(real_roots > s_min) & (real_roots < s_max)]
    if in_range.size == 0:
        return ExtremumResult(
            np.nan, np.nan, np.nan, False, "No derivative roots in range"
        )
    ddcoeffs = P.polyder(dcoeffs)
    dips = []
    peaks = []
    for r in in_range:
        second = P.polyval(r, ddcoeffs)
        if second > tol_second:  # local minimum
            if dip_window is None or (dip_window[0] < r < dip_window[1]):
                dips.append(r)
        elif second < -tol_second:  # local maximum
            if peak_window is None or (peak_window[0] < r < peak_window[1]):
                peaks.append(r)
    if len(dips) != 1 or len(peaks) != 1:
        return ExtremumResult(
            np.nan, np.nan, np.nan, False, f"Found dips={dips}, peaks={peaks}"
        )
    dip = dips[0]
    peak = peaks[0]
    return ExtremumResult(dip, peak, 0.5 * (dip + peak), True)


def linear_point(dip: float, peak: float) -> float:
    return 0.5 * (dip + peak)


def _poly_second_derivative(coeffs: np.ndarray, s: float) -> float:
    """Evaluate second derivative f''(s) for polynomial with ascending coeffs."""
    k = np.arange(coeffs.size)
    mask = k >= 2
    if not np.any(mask):
        return 0.0
    return float(np.sum(k[mask] * (k[mask] - 1) * coeffs[mask] * s ** (k[mask] - 2)))


def _analytic_root_gradient(
    coeffs: np.ndarray, root: float, flat_tol: float = 1e-12
) -> np.ndarray:
    """Analytic gradient dr/da_m for root of f'(s)=0: dr/da_m = - m r^{m-1} / f''(r)."""
    f2 = _poly_second_derivative(coeffs, root)
    if abs(f2) < flat_tol:
        return np.full_like(coeffs, np.nan)
    k = np.arange(coeffs.size)
    g = np.zeros_like(coeffs)
    mask = k >= 1
    g[mask] = -k[mask] * root ** (k[mask] - 1) / f2
    return g


@dataclass
class LPUncertainty:
    sigma_dip: float
    sigma_peak: float
    sigma_lp: float
    grad_dip: np.ndarray
    grad_peak: np.ndarray
    grad_lp: np.ndarray


def propagate_lp_uncertainty(
    coeffs: np.ndarray,
    param_cov_full: np.ndarray,
    dip: float,
    peak: float,
) -> LPUncertainty:
    """Analytic uncertainty propagation for dip / peak / Linear Point.

    Uses implicit function derivative of f'(s)=0: dr/da_m = - m r^{m-1} / f''(r).
    Returns NaN uncertainties if f'' is (near) zero at extrema (flat inflection).
    """
    g_dip = _analytic_root_gradient(coeffs, dip)
    g_peak = _analytic_root_gradient(coeffs, peak)
    g_lp = 0.5 * (g_dip + g_peak)
    if np.any(np.isnan(g_dip)) or np.any(np.isnan(g_peak)):
        return LPUncertainty(np.nan, np.nan, np.nan, g_dip, g_peak, g_lp)
    sigma_dip = float(np.sqrt(g_dip @ param_cov_full @ g_dip))
    sigma_peak = float(np.sqrt(g_peak @ param_cov_full @ g_peak))
    sigma_lp = float(np.sqrt(g_lp @ param_cov_full @ g_lp))
           
    return LPUncertainty(sigma_dip, sigma_peak, sigma_lp, g_dip, g_peak, g_lp)


@dataclass
class RealizationResult:
    chi2: float
    dof: int
    dip: float
    peak: float
    lp: float
    sigma_dip: float
    sigma_peak: float
    sigma_lp: float
    success: bool
    message: str = ""


def process_realization(
    s: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray,
    order: int,
    dip_window: Optional[tuple] = None,
    peak_window: Optional[tuple] = None,
    model_mask: Optional[Sequence[bool]] = None,
    cholesky: bool = False,
) -> RealizationResult:
    """Run full pipeline for one mock realization.

    Returns dip/peak/LP and uncertainties. Failures (e.g. missing extrema) marked success=False.
    """
    model = PolynomialModel(order=order, active_mask=model_mask)
    if cholesky:
        fit = gls_polynomial_fit_cholesky(s, y, covariance, model)
    else: 
        fit = gls_polynomial_fit(s, y, covariance, model)

    if not fit.success:
        return RealizationResult(
            np.nan,
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            fit.message,
        )
    coeffs = fit.full_coeffs
    ext = find_dip_peak(coeffs, float(s.min()), float(s.max()), dip_window, peak_window)
    if not ext.success:
        return RealizationResult(
            fit.chi2,
            fit.dof,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            ext.message,
        )
    # Build full param covariance (inactive params variance = 0)
    full_cov = np.zeros((coeffs.size, coeffs.size))
    mask = np.array(model.active_mask, dtype=bool)
    full_cov[np.ix_(mask, mask)] = fit.param_cov
    unc = propagate_lp_uncertainty(coeffs, full_cov, ext.dip, ext.peak)
    # If any uncertainty is NaN, treat this realization as a failure
    if not np.all(np.isfinite([unc.sigma_dip, unc.sigma_peak, unc.sigma_lp])):
        return RealizationResult(
            fit.chi2,
            fit.dof,
            ext.dip,
            ext.peak,
            ext.lp,
            unc.sigma_dip,
            unc.sigma_peak,
            unc.sigma_lp,
            False,
            "NaN uncertainty (sigma_dip/peak/lp)",
        )
    return RealizationResult(
        fit.chi2,
        fit.dof,
        ext.dip,
        ext.peak,
        ext.lp,
        unc.sigma_dip,
        unc.sigma_peak,
        unc.sigma_lp,
        True,
        "",
    )


@dataclass
class AggregateResults:
    n_total: int
    n_success: int
    fail_rate: float
    mean_lp: float
    std_lp: float
    mean_sigma_lp: float
    bias_over_sigma: float
    ks_lp_pvalue: float
    ks_chi2_pvalue: float
    chi2_reduced_mean: float
    lp_values: np.ndarray
    sigma_lp_values: np.ndarray


def aggregate_lp_statistics(
    results: Sequence[RealizationResult],
    fiducial_lp: Optional[float] = None,
) -> AggregateResults:
    successes = [r for r in results if r.success]
    lp_vals = np.array([r.lp for r in successes])
    sigma_lp_vals = np.array([r.sigma_lp for r in successes])
    mean_lp = float(lp_vals.mean()) if lp_vals.size else float("nan")
    std_lp = float(lp_vals.std(ddof=1)) if lp_vals.size > 1 else float("nan")
    mean_sigma_lp = float(sigma_lp_vals.mean()) if sigma_lp_vals.size else float("nan")
    if fiducial_lp is not None and std_lp and not np.isnan(std_lp):
        bias_over_sigma = (mean_lp - fiducial_lp) / std_lp
    else:
        bias_over_sigma = float("nan")
    # KS test vs normal using standardized values
    if lp_vals.size > 3 and std_lp and not np.isnan(std_lp):
        standardized = (lp_vals - mean_lp) / std_lp
        ks_stat, ks_p = stats.kstest(standardized, "norm")
    else:
        ks_p = float("nan")
    chi2_reduced = [r.chi2 / r.dof for r in successes if r.dof > 0]
    # KS test for chi^2 distribution (requires consistent dof across realizations)
    chi2_vals = np.array([r.chi2 for r in successes if r.dof > 0])
    dofs = np.array([r.dof for r in successes if r.dof > 0])
    if chi2_vals.size > 3 and np.all(dofs == dofs[0]):
        try:
            _, ks_chi2_p = stats.kstest(chi2_vals, "chi2", args=(int(dofs[0]),))
        except Exception:
            ks_chi2_p = float("nan")
    else:
        ks_chi2_p = float("nan")
    chi2_reduced_mean = float(np.mean(chi2_reduced)) if chi2_reduced else float("nan")
    return AggregateResults(
        n_total=len(results),
        n_success=len(successes),
        fail_rate=1 - len(successes) / len(results) if results else float("nan"),
        mean_lp=mean_lp,
        std_lp=std_lp,
        mean_sigma_lp=mean_sigma_lp,
        bias_over_sigma=bias_over_sigma,
        ks_lp_pvalue=ks_p,
        ks_chi2_pvalue=ks_chi2_p,
        chi2_reduced_mean=chi2_reduced_mean,
        lp_values=lp_vals,
        sigma_lp_values=sigma_lp_vals,
    )


if __name__ == "__main__":  # pragma: no cover - simple smoke test
    # Tiny self-test (not exhaustive)
    s = np.linspace(80, 110, 5)
    xi_vals = np.vstack([s, np.sin(s / 10) * 1e-3]).T  # (N,2)
    # Build a toy covariance with modest correlations
    N = xi_vals.shape[0]
    base = 1e-6 * np.exp(-0.5 * ((s[:, None] - s[None, :]) / 5) ** 2)
    result = create_random_xi_data(xi_vals, base, n_samples=10, seed=123)
    print("Generated samples shape:", result.samples.shape)
