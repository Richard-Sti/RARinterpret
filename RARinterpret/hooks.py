# Copyright (C) 2023 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Functions that attempt to straighten hooks in the SPARC RAR.
"""
from jax import numpy as jnp
from jax.scipy.special import erfc


###############################################################################
#            JAX-compatible Simpson's rule numerical integration              #
###############################################################################


def simps(f, a, b, N=128, *args):
    """
    Simpson's rule for numerical integration of `f(x)` from `a` to `b` using
    `N` uniformly spaced points between `a` and `b`.

    Parameters
    ----------
    f : function
        Function `f(x)` to integrate.
    a, b : float
        Integration limits.
    N : int, optional
        Number of points to use. Must be an even integer. The default is 128.
    *args : tuple
        Additional arguments to pass to `f`.

    Returns
    -------
    float
    """
    if N % 2 != 0:
        raise ValueError("`N` must be an even integer.")

    x = jnp.linspace(a, b, N + 1)
    y = f(x, *args)
    dx = (b - a) / N
    return dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


###############################################################################
#                        Probability that x2 > x1                             #
###############################################################################


def x2_larger_than_x1(mu1, std1, mu2, std2):
    """
    Determine the probability that a Gaussian random variable `x2` with mean
    `mu2` and standard deviation `std2` is larger than a Gaussian random
    variable `x1` with mean `mu1` and standard deviation `std1`.

    Parameters
    ----------
    mu1 : float
        Mean of `x1`.
    std1 : float
        Standard deviation of `x1`.
    mu2 : float
        Mean of `x2`.
    std2 : float
        Standard deviation of `x2`.

    Returns
    -------
    float
    """
    def integrand(x):
        return jnp.exp(-0.5 * x**2)  * erfc((x * std1 + mu1 - mu2) / (jnp.sqrt(2) * std2))  # noqa

    return simps(integrand, -15, 15, 512) / (2 * jnp.pi) * (jnp.pi / 2)**0.5


def is_increasing(y, yerr):
    """
    Calculate the log-probability that the values in `y` are monotonically
    increasing. Assumes that the values in `y` are Gaussian random variables
    whose means are the values in `y` and whose standard deviations are the
    values in `yerr`.

    Parameters
    ----------
    y : 1-dimensional array
        Array of values.
    yerr : 1-dimensional array
        Array of uncertainties.

    Returns
    -------
    float
    """
    lprob = 0.0
    for i in range(1, len(y)):
        for j in range(i):
            lprob += jnp.log10(x2_larger_than_x1(y[j], yerr[j], y[i], yerr[i]))

    return lprob
