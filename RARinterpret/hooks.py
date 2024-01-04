# Copyright (C) 2023 Richard Stiskalek, Harry Desmond
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
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import jit
from jax import numpy as jnp
from jax.debug import print as jprint  # noqa
from jax.scipy.special import erfc
from tqdm import trange

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

    return simps(integrand, -15, 15, 256) / (2 * jnp.pi) * (jnp.pi / 2)**0.5


def _is_increasing(y, yerr):
    lprob = 0.0
    for i in range(1, len(y)):
        for j in range(i):
            lprob += jnp.log10(x2_larger_than_x1(y[j], yerr[j], y[i], yerr[i]))

    return lprob


def is_increasing(y, yerr, return_log=True, normalize=False):
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
    return_log : bool
        Whether to return the log-probability or the probability.
    normalize : bool
        Whether to normalize the probability by taking it to the power of
        `1 / len(y)`.

    Returns
    -------
    float
    """
    lprob = _is_increasing(y, yerr)

    if normalize:
        lprob /= len(y)

    if return_log:
        return lprob

    return 10**lprob


###############################################################################
#                   NumPyro code to straighten a single hook                  #
###############################################################################


def lognormal_mean_std_to_loc_shape(mean, std):
    """
    Convert the mean and standard deviation of a lognormal distribution to its
    location and shape parameters.

    Parameters
    ----------
    mean : float
        Mean of the lognormal distribution.
    std : float
        Standard deviation of the lognormal distribution.

    Returns
    -------
    loc, shape : floats
    """
    loc = np.log(mean) - 0.5 * np.log(1 + std**2 / mean**2)
    shape = np.log(1 + std**2 / mean**2)**0.5
    return loc, shape


class AccelerationRotationCurveModel:
    """
    NumPyro model for fitting a single galaxy's rotation curve in the
    log-gbar vs log-gobs plane.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy.
    frame : RARFrame
        RARFrame object.
    intrinsic_scatter : bool, optional
        Whether to include intrinsic scatter in the model. The default is True.
    monotonicity : bool, optional
        Whether to enforce monotonicity in the model. The default is False.
    """
    _intrinsic_scatter = None
    _monotonocity = None

    def __init__(self, galaxy_name, frame, intrinsic_scatter=True,
                 monotonicity=False):
        try:
            gindx = frame._name2gindx[galaxy_name]
        except KeyError:
            raise ValueError(f"Galaxy {galaxy_name} not found.")

        mask = frame["index"] == gindx

        gobs = frame["gobs"][mask]
        Vobs = frame["Vobs"][mask]
        e_Vobs = frame["e_Vobs"][mask]
        e_gobs_stat = 2 * e_Vobs / Vobs

        log_gobs = (np.log(gobs) - 0.5 * np.log(1 + e_gobs_stat**2)) / np.log(10.)  # noqa
        e_log_gobs = np.sqrt(np.log(1 + e_gobs_stat**2)) / np.log(10.)              # noqa

        Vgas = frame["Vgas"][mask]
        Vdisk = frame["Vdisk"][mask]
        Vbul = frame["Vbul"][mask]

        self._data = {
            "gobs": gobs,
            "e_gobs_stat": e_gobs_stat,
            "log_gobs": log_gobs,
            "e_log_gobs": e_log_gobs,
            "e_log_gobs2": e_log_gobs**2,
            "gbar": frame["gbar"][mask],
            "r": frame["r"][mask],
            "dist": frame["dist"][mask][0],
            "e_dist": frame["e_dist"][mask][0],
            "inc": frame["inc"][mask][0] * np.pi / 180,
            "e_inc": frame["e_inc"][mask][0] * np.pi / 180,
            "L36": frame["L36"][mask][0],
            "e_L36": frame["e_L36"][mask][0],
            "Vgas2": Vgas * np.abs(Vgas),
            "Vdisk2": Vdisk * np.abs(Vdisk),
            "Vbul2": Vbul * np.abs(Vbul),
        }

        self.intrinsic_scatter = intrinsic_scatter
        self._monotonocity = monotonicity

    @property
    def intrinsic_scatter(self):
        """
        Whether to include intrinsic scatter in the model.

        Returns
        -------
        bool
        """
        if self._intrinsic_scatter is None:
            raise ValueError("Intrinsic scatter flag is not set.")
        return self._intrinsic_scatter

    @intrinsic_scatter.setter
    def intrinsic_scatter(self, value):
        if not isinstance(value, bool):
            raise TypeError("Intrinsic scatter flag must be a boolean.")
        self._intrinsic_scatter = value

    @property
    def monotonicity(self):
        """
        Whether to enforce monotonicity in the model.

        Returns
        -------
        bool
        """
        if self._monotonocity is None:
            raise ValueError("Monotonicity flag is not set.")
        return self._monotonocity

    @monotonicity.setter
    def monotonicity(self, value):
        if not isinstance(value, bool):
            raise TypeError("Monotonicity flag must be a boolean.")
        self._monotonocity = value

    def scale_gbar(self, ML_gas, ML_disk, ML_bul, L36):
        """
        Scale `gbar` from the observed fiducial values to the 'true' values.

        Parameters
        ----------
        ML_gas : float
            Gas mass-to-light ratio.
        ML_disk : float
            Disk mass-to-light ratio.
        ML_bul : float
            Bulge mass-to-light ratio.
        L36 : float
            Luminosity at 3.6 microns.

        Returns
        -------
        gbar : 1-dimensional array
            Scaled `gbar` values in units of `1e-10 m/s^2`.
        """
        Vbar2 = ML_gas * self["Vgas2"] + L36 / self["L36"] * (ML_disk * self["Vdisk2"] + ML_bul * self["Vbul2"])  # noqa
        gbar = 1.e10 * Vbar2 / self["r"] * 1.e6 / 3.08567758e19
        return gbar

    def scale_gobs(self, inc, dist):
        """
        Scale `gobs` from the observed fiducial values to the 'true' values.

        Parameters
        ----------
        inc : float
            Inclination in radians.
        dist : float
            Distance in kpc.

        Returns
        -------
        gobs : 1-dimensional array
            Scaled `gobs` values in units of `1e-10 m/s^2`.
        """
        return self["gobs"] * (jnp.sin(self["inc"]) / jnp.sin(inc))**2 * (self["dist"] / dist)  # noqa

    def scale_log_gobs(self, inc, dist):
        """
        Scale `log_gobs` from the observed fiducial values to the 'true'
        values, while accounting for the statistical uncertainty in `gobs`.

        Parameters
        ----------
        inc : float
            Inclination in radians.
        dist : float
            Distance in kpc.

        Returns
        -------
        log_gobs : 1-dimensional array
            Scaled `log_gobs` values.
        """
        gobs = self.scale_gobs(inc, dist)
        log_gobs = (jnp.log(gobs) - 0.5 * jnp.log(1 + self["e_gobs_stat"]**2)) / jnp.log(10.)  # noqa
        return log_gobs

    def make_model_from_samples(self, samples):
        """
        Make log-gbar and log-gobs arrays from a dictionary of samples.

        Parameters
        ----------
        samples : dict
            Dictionary of samples from a NumPyro model.

        Returns
        -------
        log_gbar : 2-dimensional array of shape (nsamples, nrc)
            Log-gbar values.
        log_gobs : 2-dimensional array of shape (nsamples, nrc)
            Log-gobs values.
        """
        nsamples = len(samples["L36"])
        log_gbar = np.zeros((nsamples, len(self)))
        log_gobs = np.zeros((nsamples, len(self)))

        for i in range(nsamples):
            log_gbar[i] = np.log10(self.scale_gbar(
                samples["ML_gas"][i],
                samples["ML_disk"][i],
                samples["ML_bul"][i],
                samples["L36"][i],
            ))
            log_gobs[i] = self.scale_log_gobs(
                samples["inc"][i],
                samples["dist"][i],
            )

        return log_gbar, log_gobs

    def evaluate_monotonocity(self, log_gbar, log_gobs, scatter):
        if log_gobs.ndim == 1 or log_gbar.ndim == 1:
            log_gobs = log_gobs.reshape(1, -1)
            log_gbar = log_gbar.reshape(1, -1)
            if not isinstance(scatter, (float, int)):
                raise TypeError("Scatter must be a float or int.")
            scatter = [scatter]

        f = jit(_is_increasing)

        nsamples = len(log_gobs)
        out = np.zeros(nsamples)
        for n in trange(nsamples, desc="Monotonicity"):
            yerr = np.sqrt(scatter[n]**2 + self["e_log_gobs2"])
            out[n] = f(log_gobs[n], yerr)

        if log_gobs.size == 1:
            out = out[0]

        return out

    def rotation_curve_model(self):
        """
        Generate a NumPyro template model for the galaxy's rotation curve.
        """
        # Sample hyperparameters
        if self.intrinsic_scatter:
            scatter = numpyro.sample("scatter", dist.TruncatedNormal(0.0335, 0.001, low=0))                     # noqa
        a0 = numpyro.sample("a0", dist.Normal(1.215, 0.1))

        ML_disk = numpyro.sample("ML_disk", dist.LogNormal(*lognormal_mean_std_to_loc_shape(0.5, 0.125)))       # noqa
        ML_bul = numpyro.sample("ML_bul", dist.LogNormal(*lognormal_mean_std_to_loc_shape(0.7, 0.175)))         # noqa
        ML_gas = numpyro.sample("ML_gas", dist.LogNormal(*lognormal_mean_std_to_loc_shape(1., 0.1)))            # noqa
        d = numpyro.sample("dist", dist.TruncatedNormal(self["dist"], self["e_dist"], low=0))                   # noqa
        inc = numpyro.sample("inc", dist.TruncatedNormal(self["inc"], self["e_inc"], low=0, high=np.pi / 2))    # noqa
        L36 = numpyro.sample("L36", dist.TruncatedNormal(self["L36"], self["e_L36"], low=0))                    # noqa

        # Scale gbar and log(gobs)
        gbar = self.scale_gbar(ML_gas, ML_disk, ML_bul, L36)
        log_gobs = self.scale_log_gobs(inc, d)

        # Calculate the predicted log(gobs) from the RAR
        mu = gbar / (1 - jnp.exp(- jnp.sqrt(gbar / a0)))
        log_mu = jnp.log10(mu)

        if self.intrinsic_scatter:
            yerr = jnp.sqrt(scatter**2 + self["e_log_gobs2"])
        else:
            yerr = self["e_log_gobs"]

        log_likelihood = dist.Normal(log_mu, yerr).log_prob(log_gobs)

        if self.monotonicity:
            # NOTE: Should these be sorted? Currently sorted by radius.
            alpha = _is_increasing(log_gobs, yerr)
            numpyro.factor("obs", jnp.sum(log_likelihood) + alpha)
        else:
            numpyro.factor("obs", jnp.sum(log_likelihood))

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data["gobs"])
