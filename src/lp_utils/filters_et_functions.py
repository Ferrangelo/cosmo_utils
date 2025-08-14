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

deriv_spherical_bessel_j(n, x)
    Computes the derivative of the spherical Bessel function of the first kind j_n(x) for the given order and input value(s).

dj0(x)
    Computes the derivative of the spherical Bessel function of the first kind j_0(x) for the input value(s).
"""

import numpy as np
from scipy.special import spherical_jn


def top_hat_filter(k, R):
    """
    Computes the top-hat filter in Fourier space.
    It uses the spherical Bessel function of the first kind to compute the filter.

    Parameters
    ----------
    k : array_like
        Array of wavenumbers.
    R : float
        Radius of the top-hat filter.

    Returns
    -------
    f : array_like
        The top-hat filter values corresponding to the input wavenumbers `k`.
    """
    x = k * R
    x = np.asarray(x)
    result = np.ones_like(x, dtype=float)
    mask = x != 0
    result[mask] = 3 * spherical_jn(1, x[mask]) / x[mask]
    return result


def wgc(x, xpiv=1, n=4):
    """
    Weighted Gaussian-like filter used to avoid ringing when computing 2 point correlation frunction from pk.

    Parameters
    ----------
    x : array_like or float
        Input value(s) at which to evaluate the function.
    xpiv : float, optional
        Pivot value for normalization (default is 1).
    n : int or float, optional
        Exponent controlling the sharpness of the curve (default is 4).

    Returns
    -------
    float or ndarray
        The computed value(s) of the weighted Gaussian-like curve.

    Notes
    -----
    The function is defined as: exp(-((x / xpiv) ** n)).
    """
    return np.exp(-((x / xpiv) ** n))


def j0(x):
    """
    Compute the spherical Bessel function of the first kind of order 0 for input x.

    Parameters:
        x (float or array-like): Input value(s).

    Returns:
        float or ndarray: The value(s) of the spherical Bessel function j0 at x.
    """
    return spherical_jn(0, x)


def deriv_spherical_bessel_j(n, x):
    """
    Computes the derivative of the spherical Bessel function of the first kind j_n(x).

    Parameters
    ----------
    n : int
        Order of the spherical Bessel function.
    x : float or array-like
        Input value(s) at which to compute the derivative.

    Returns
    -------
    float or ndarray
        The derivative of the spherical Bessel function j_n(x) at the input value(s).
    """
    if np.any(x == 0):
        raise ValueError(
            "This function does not accept zero values for x. Please handle zero values separately when calling this function."
        )

    return n * spherical_jn(n, x) / x - spherical_jn(n + 1, x)


def dj0(x):
    """
    Computes the derivative of the spherical Bessel function of the first kind j_0(x).

    Parameters
    ----------
    x : float or array-like
        Input value(s) at which to compute the derivative.

    Returns
    -------
    float or ndarray
        The derivative of the spherical Bessel function j_0(x) at the input value(s).

    Notes
    -----
    - Converts the input `x` to a NumPy array to ensure consistent array operations.
    - Initializes an array `result` of zeros with the same shape as `x` to store the output.
    - Creates a boolean mask to identify elements where `x` is not zero, since the derivative is zero at zero.
    - Computes the derivative only for nonzero elements using `deriv_spherical_bessel_j` to avoid division by zero.
    - Returns the resulting array, which contains the derivative values at the specified input points.

    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask = x != 0
    result[mask] = deriv_spherical_bessel_j(0, x[mask])
    return result
