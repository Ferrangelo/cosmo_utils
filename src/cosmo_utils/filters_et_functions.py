"""
This module provides a set of filter and special function utilities.

Functions
---------
top_hat_filter(k, R)
    Computes the top-hat filter in Fourier space using the spherical Bessel function of the first kind.

wgc(x, xpiv=1, n=4)
    Weighted Gaussian-like filter to avoid ringing when computing the two-point correlation function from the power spectrum.

j0(x)
    Computes the spherical Bessel function of the first kind of order 0 for the input value(s).

j1(x)
    Computes the spherical Bessel function of the first kind of order 1 for the input value(s).

dj0(x)
    Computes the derivative of the spherical Bessel function of the first kind j_0(x) for the input value(s).

d_dr_j0_of_kr(k, r)
    Computes the derivative of the spherical Bessel function of the first kind j_0(kr) with respect to r.
"""

import numpy as np
from scipy.special import spherical_jn


def top_hat_filter(k, R):
    """
    Computes the top-hat filter in Fourier space using spherical Bessel j1.

    Parameters
    ----------
    k : array_like  Wavenumbers.
    R : float  Filter radius.

    Returns
    -------
    array_like  Top-hat filter values at k.
    """
    x = k * R
    x = np.asarray(x)
    result = np.ones_like(x, dtype=float)
    mask = x != 0
    result[mask] = 3 * spherical_jn(1, x[mask]) / x[mask]
    return result


def wgc(x, xpiv=1, n=4):
    """
    Weighted Gaussian-like filter exp(-((x/xpiv)^n)) to suppress ringing.

    Parameters
    ----------
    x : array_like or float  Input values.
    xpiv : float, optional  Pivot normalization (default 1).
    n : int or float, optional  Exponent controlling sharpness (default 4).

    Returns
    -------
    float or ndarray  Filter values.
    """
    return np.exp(-((x / xpiv) ** n))


def j0(x):
    """
    Spherical Bessel function j0(x).

    Parameters
    ----------
    x : float or array-like  Input value(s).

    Returns
    -------
    float or ndarray  j0(x).
    """
    return spherical_jn(0, x)


def j1(x):
    """
    Spherical Bessel function j1(x).

    Parameters
    ----------
    x : float or array-like  Input value(s).

    Returns
    -------
    float or ndarray  j1(x).
    """
    return spherical_jn(1, x)


def dj0(x):
    """
    Derivative of spherical Bessel j0(x).

    Parameters
    ----------
    x : float or array-like  Input value(s).

    Returns
    -------
    float or ndarray  d/dx j0(x).
    """
    return spherical_jn(0, x, derivative=True)


def d_dr_j0_of_kr(k, r):
    """
    Derivative with respect to r of j0(k r).

    Parameters
    ----------
    k : float or array-like  Wavenumber(s).
    r : float or array-like  Radius value(s).

    Returns
    -------
    float or ndarray  d/dr j0(k r).
    """
    return k * dj0(k * r)


def j0_bar(k, ri, delta_r):
    """
    Computes the average value of the spherical Bessel function j0 over a spherical shell of radius ri and thickness delta_r.
    Supports k (1D), ri (scalar, 1D, or 2D).
    """
    from cosmo_utils.cosmology import Vs
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


if __name__ == "__main__":
    print(
        "This module provides filter and special function utilities and is not intended to be run directly."
    )
