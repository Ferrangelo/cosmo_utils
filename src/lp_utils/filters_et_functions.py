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

def j1(x):
    """
    Compute the spherical Bessel function of the first kind of order 1 for input x.

    Parameters:
        x (float or array-like): Input value(s).

    Returns:
        float or ndarray: The value(s) of the spherical Bessel function j1 at x.
    """
    return spherical_jn(1, x)



def dj0(x):
    """
    Computes the derivative of the spherical Bessel function of the first kind j_0(x).

    Parameters:
        x (float or array-like): Input value(s).

    Returns:
        float or ndarray: The derivative of the spherical Bessel function j_0 at x.
    """
    return spherical_jn(0, x, derivative=True)

def d_dr_j0_of_kr(k, r):
    """
    Computes the derivative of the spherical Bessel function of the first kind j_0(kr) with respect to r.

    Parameters:
        k (float or array-like): Wavenumber(s).
        r (float or array-like): Radius(es).

    Returns:
        float or ndarray: The derivative of j_0(kr) with respect to r.
    """
    return k * dj0(k * r)


if __name__ == "__main__":
    print(
        "This module provides filter and special function utilities and is not intended to be run directly."
    )
