# Copyright (C) 2022 Richard Stiskalek
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
The RAR interpolator and ML modelling functions.
"""
from abc import ABC, abstractmethod
from functools import partial

import jax
import numpy
from astropy import units
from scipy.special import i0, i1, k0, k1
from jax import numpy as jnp

KPC2KM = units.kpc.to(units.km)


class Base1D(ABC):
    """
    Base class for fitting 1D functions.
    """
    _dydx = None
    _use_biased = True
    _with_scatter = None
    _guess_logscatter = None

    def __init__(self, use_biased=False, with_scatter=False,
                 guess_logscatter=-1):
        dydx = (jax.grad(self.predict, argnums=(0,)))
        self._dydx = jax.jit(jax.vmap(dydx, in_axes=(0, None)))
        self.use_biased = use_biased
        self.with_scatter = with_scatter
        self.guess_logscatter = guess_logscatter

    @property
    def use_biased(self):
        """
        Whether to use the biased likelihood.

        Returns
        -------
        use_biased : bool
        """
        return self._use_biased

    @use_biased.setter
    def use_biased(self, use_biased):
        """Set `use_biased`."""
        assert isinstance(use_biased, bool)
        self._use_biased = use_biased

    @property
    def with_scatter(self):
        """
        Whether to add intrinsic scatter term in the likelihood.

        Returns
        -------
        with_scatter : bool
        """
        return self._with_scatter

    @with_scatter.setter
    def with_scatter(self, with_scatter):
        """Set `with_scatter`."""
        assert isinstance(with_scatter, bool)
        self._with_scatter = with_scatter

    @property
    def guess_logscatter(self):
        """
        Initial guess for the logarithmic intrinsic scatter.

        Returns
        -------
        guess_logscatter : foat
        """
        return self._guess_logscatter

    @guess_logscatter.setter
    def guess_logscatter(self, guess_logscatter):
        """Set `guess_logscatter`."""
        assert isinstance(guess_logscatter, (float, int))
        self._guess_logscatter = guess_logscatter

    @property
    def nparams(self):
        """
        Number of function parameters. If `with_scatter` accounts for the
        scatter term too.

        Returns
        -------
        nparams : int
        """
        if not hasattr(self, "_nparams"):
            raise ValueError("`n_params` must be set in the class definition!")
        return self._nparams + int(self.with_scatter)

    @property
    def x0(self):
        """
        A good initial function parameter guess. Appends the intrinsic scatter
        at the last position if `with_scatter` is `True`.

        Returns
        -------
        x0 : 1-dimensional array
        """
        if not hasattr(self, "_x0"):
            raise ValueError("`x0` must be set in the class definition!")
        if self.with_scatter:
            return numpy.concatenate([self._x0, [self.guess_logscatter]])
        return self._x0

    @x0.setter
    def x0(self, x0):
        raise NotImplementedError

    def predict(self, x, thetas):
        r"""
        Predicted value.

        Parameters
        ----------
        x : 1-dimensional array
            Values of the independant variable.
        thetas : 1-dimensional array
            Function parameters.

        Returns
        -------
        log_gobs : 1-dimensional array
        """
        if thetas.size != self.nparams:
            raise TypeError("Size of `thetas` must match `nparams`!")
        return self._predict(x, thetas[:-1] if self.with_scatter else thetas)

    @abstractmethod
    def _predict(self, x, thetas):
        pass

    def dpredict(self, x, thetas):
        r"""
        Gradient of the function with respect to `x`.

        Parameters
        ----------
        x : 1-dimensional array
            Values of the independent variable.
        thetas : 1-dimensional array
            Function parameters.

        Returns
        -------
        grad : 1-dimensional array
        """
        return self._dydx(x, thetas)[0]

    def make_weights(self, thetas, x, varx, vary, dydx=None):
        r"""
        Calculate weights, defined as the inverse variance

        ..math:

            \sigma_{y}^2 + \left(\frac{\dd y}{\dd x} \sigma_{x}\right)^2.

        The weigts are optionally normalised so that their sum is unity.

        Parameters
        ----------
        thetas : 1-dimensional array
            Function parameters.
        x : 1-dimensional array
            Values of the independent variable `x`.
        varx : 1-dimensional array
            Variance of `x`.
        vary : 1-dimensional array
            Variance of `y`, the dependent variable.
        dydx : 1-dimensional array, optional
            Optional precomputed gradient, if provided replaces the AD
            gradient.

        Returns
        -------
        weights : 1-dimensional array
            Weights.
        """
        if dydx is None:
            dydx = self.dpredict(x, thetas)
        var = vary + jnp.square(dydx) * varx
        return 1 / var

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, thetas, x, y, varx, vary, dydx=None):
        r"""
        Loss function.

        Parameters
        ----------
        thetas : 1-dimensional array
            Function parameters.
        x : 1-dimensional array
            Values of the independent variable `x`.
        y : 1-dimensional array
            Values of the dependent variable `y`.
        varx : 1-dimensional array
            Variance of `x`.
        vary : 1-dimensional array
            Variance of `y`, the dependent variable.
        dydx : 1-dimensional array, optional
            Optional precomputed gradient, if provided replaces the AD
            gradient.

        Returns
        -------
        loss : float
        """
        ypred = self.predict(x, thetas)
        if dydx is None:
            dydx = self.dpredict(x, thetas)
        var = vary + jnp.square(dydx) * varx
        var += self.with_scatter * 10**(2 * thetas[-1])
        loss = 0.5 * jnp.sum(jnp.square(ypred - y) / var)
        # Optionally add the biased likelihood contribution
        loss += self.use_biased * 0.5 * jnp.sum(jnp.log(2 * numpy.pi * var))
        return loss


class RARIF(Base1D):
    r"""
    Class to fit the RAR as specified in Eq. 4 of [1]. Both the input and
    output are in :math:`\log_{10}`.

    References
    ----------
    [1] The Radial Acceleration Relation in Rotationally Supported Galaxies;
    Stacy McGaugh, Federico Lelli, Jim Schombert
    """
    _nparams = 1
    name = "IF"
    _x0 = numpy.asanyarray([1.118])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        log_a0 = jnp.log10(thetas[0])
        ratio = jnp.power(10, x - log_a0)
        return x - jnp.log10(1 - jnp.exp(- jnp.sqrt(ratio)))

    def VbarVobs2r(self, Vbar, Vobs, thetas):
        r"""
        Calculate :math:`r` from :math:`V_{\rm bar}` and :math:`V_{\rm obs}`
        using the RAR IF.

        Parameters
        ----------
        Vbar, Vobs : 1-dimensional arrays
            Baryonic and observed rotational velocity in
            :math:`\mathrm{km} \mathrm{s}^{-1}`
        thetas : 1-dimensional array
            Function parameters.

        Returns
        -------
        r : 1-dimensional array
            Galactocentric distance in :math:`\mathrm{kpc}`.
        """
        a0 = thetas[0]
        r = Vbar**2 / a0 / numpy.log(1 - (Vbar / Vobs)**2)**2
        r *= 1e13 / KPC2KM
        return r

    def rVbar2Vobs(self, r, Vbar, thetas):
        r"""
        Calculate :math:`V_{\rm obs}` from :math:`V_{\rm bar}` and :math:`r`
        using the RAR IF.

        Parameters
        ----------
        r, Vbar: 1-dimensional arrays
            Galactocentric distance in :math:`\mathrm{kpc}` and baryonic
            rotational velocity in :math:`\mathrm{km} \mathrm{s}^{-1}`.
        thetas : 1-dimensional array
            Function parameters.

        Returns
        -------
        Vobs : 1-dimensional array
            Obsered rotational velocity in :math:`\mathrm{km} \mathrm{s}^{-1}`.
        """
        a0 = thetas[0]
        gbar = Vbar**2 / r * 1e13 / KPC2KM
        return Vbar / numpy.sqrt(1 - numpy.exp(-numpy.sqrt(gbar / a0)))


class SimpleIFEFE(Base1D):
    r"""
    The AQUAL RAR EFE function, for definition see Eq. 6 of [1]. Both the input
    and output are in :math:`\log_{10}`.

    References
    ----------
    [1] On the functional form of the radial acceleration relation;
    Harry Desmond, Deaglan J. Bartlett, Pedro G. Ferreira (2023)
    """
    _nparams = 2
    name = "EFE"
    _x0 = numpy.asanyarray([1.169, -2.09312646527793])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        a0, a1 = thetas
        x = jnp.power(10., x)
        a1 = jnp.power(10., a1)
        y = x*(0.5 + jnp.sqrt(0.25 + ((x/a0)**2 + (1.1*a1)**2)**(-1/2)))*(1 + jnp.tanh(1.1*a1/(x/a0))**1.2 * (-1/3) * (1/2*((x/a0)**2 + (1.1*a1)**2)**(-1/2) * (0.25 + ((x/a0)**2 + (1.1*a1)**2)**(-1/2))**(-1/2)/(0.5 + (0.25 + ((x/a0)**2 + (1.1*a1)**2)**(-1/2))**0.5)))  # noqa
        return jnp.log10(y)


class BestRARESR(Base1D):
    r"""
    Symbolic regression RAR inferred function from [1]:

    ..math:
        g_{\rm obs} = ||\theta_1|^x + \theta_0|^{\theta_2} + x,

    where :math:`x = g_{\rm bar}`. Both the input and output are in
    :math:`\log_{10}`.

    References
    ----------
    [1] On the functional form of the radial acceleration relation;
    Harry Desmond, Deaglan J. Bartlett, Pedro G. Ferreira (2023)
    """
    _nparams = 3
    name = "SR1"
    _x0 = numpy.asanyarray([-0.995, 0.64, 0.36])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        theta0, theta1, theta2 = thetas
        gbar = jnp.power(10., x)
        out = jnp.power(jnp.abs(theta1), gbar) + theta0
        out = jnp.abs(out)
        out = jnp.power(out, theta2) + gbar
        return jnp.log10(out)


class LinearRelation(Base1D):
    r"""
    A linear function.
    """
    _nparams = 2
    name = "Linear"
    _x0 = numpy.asanyarray([0.5, 0.0])  # Match Vbar - Vobs

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        m, c = thetas
        return x * m + c


class RVRQuadratic(Base1D):
    r"""
    A quadratic function for the :math:`\log V_{\rm bar} - \log V_{\rm obs}`
    relation.
    """
    _nparams = 3
    name = "RVRQuadratic"
    # Match Vbar - Vobs
    _x0 = numpy.asanyarray([-0.09834573,  1.12822747,  0.34722243])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        a1, a2, a3 = thetas
        return a1 * x**2 + a2 * x + a3


class LVQuadratic(Base1D):
    r"""
    A quadratic function for the :math:`\log L_{3.6} - \log V_{\rm obs}`
    relation.
    """
    _nparams = 3
    name = "LVQuadratic"
    # Match Vbar - Vobs
    _x0 = numpy.asanyarray([0.01618431, 0.21042491, 1.81068696])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        a1, a2, a3 = thetas
        return a1 * x**2 + a2 * x + a3


class SJCubic(Base1D):
    r"""
    A cubic function for the :math:`\log \Sigma - \log J_{\rm obs}` relation,
    where :math:`\Sigma = \Sigma_{\rm disk} + \Sigma_{\rm bul}`.
    """
    _nparams = 4
    name = "SJCubic"
    # Match Vbar - Vobs
    _x0 = numpy.asanyarray([0.01983164, 0.07604715, 0.268838, 4.07167671])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        a1, a2, a3, a4 = thetas
        return a1 * x**3 + a2 * x**2 + a3 * x + a4


class QuadraticRelation(Base1D):
    r"""
    A quadratic function.
    """
    _nparams = 3
    name = "Quadratic"
    # Match Vbar - Vobs
    _x0 = numpy.asanyarray([-0.09834573,  1.12822747,  0.34722243])

    @partial(jax.jit, static_argnums=(0,))
    def _predict(self, x, thetas):
        a1, a2, a3 = thetas
        return a1 * x**2 + a2 * x + a3


###############################################################################
#                            Mock data generator.                             #
###############################################################################


def rvs_positive(mean, std, seed=None):
    """
    Sample a new positive value from a Gaussian distribution.

    Parameters
    ----------
    mean, std : floats
        Mean and standard deviation.
    seed : int, optional
        Random seed.

    Returns
    -------
    dist : float
    """
    gen = numpy.random.RandomState(seed)
    n_attempts = 1000
    for __ in range(n_attempts):
        out = gen.normal(mean, std)
        if numpy.all(out > 0):
            return out

    raise RuntimeError(
        "Unable to sample positive distance for mean {} and std {}."
        .format(mean, std))


def _check_allequal(x, name):
    """
    Check that a 1-dimensional array has all elements equal.

    Parameters
    ----------
    x : 1-dimensional array
    name : string

    Returns
    -------
    None
    """
    if isinstance(x, numpy.ndarray):
        msg = "All values of {} must be equal".format(name)
        if numpy.any(numpy.isnan(x)):
            assert numpy.all(numpy.isnan(x)), msg
        else:
            assert numpy.all(x[0] == x), msg


def _check_globprob(mean_dist, e_dist, mean_inc, e_inc, mean_L36,
                    e_L36, mean_logeN, e_logeN):
    """
    Check that values in all arrays are equal and return their first
    element.

    Parameters
    ----------
    mean_dist, e_dist, mean_inc, e_inc, mean_L36,
        e_L36, mean_logeN, e_logeN : 1-dimensional arrays

    Returns
    -------
    mean_dist, e_dist, mean_inc, e_inc, mean_L36,
        e_L36, mean_logeN, e_logeN : floats
    """
    # Check that galaxy properties are consistent for a single RC
    xs = (mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36,
          mean_logeN, e_logeN)
    names = ("mean_dist", "e_dist", "mean_inc", "e_inc", "mean_L36",
             "e_L36", "mean_logeN", "e_logeN")
    for x, name in zip(xs, names):
        _check_allequal(x, name)
    mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36 = [
        x[0] for x in xs[:-2] if isinstance(x, numpy.ndarray)]

    mean_logeN = mean_logeN[0] if mean_logeN is not None else None
    e_logeN = e_logeN[0] if e_logeN is not None else None

    return (mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36,
            mean_logeN, e_logeN)


class BaseMock(ABC):
    """
    Base mock data generator. Resamples `Vbar`, `dist` and `inc` from their
    distributions.
    """

    @staticmethod
    def copy_replace_zeros(x, val):
        """
        Copy `x` and replace its zeros with `val`.

        Parameters
        ----------
        x : 1-dimensional array
        val : int or float

        Returns
        -------
        out : 1-dimensional array
        """
        out = numpy.copy(x)
        out[numpy.isclose(out, 0.)] = val
        return out

    @staticmethod
    def rvs_mass_to_light(seed=None):
        """
        Sample the gas, disk and bulge mass-to-light ratios from their
        log-normal distributions.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        ml_gas, ml_disk, ml_bulge : floats
            The gas, disk and bulge mass-to-light ratio.
        """
        gen = numpy.random.RandomState(seed)
        ml_gas = 10**gen.normal(0., 0.1 / numpy.log(10))
        ml_disk = 10**gen.normal(numpy.log10(0.5), 0.1)
        ml_bulge = 10**gen.normal(numpy.log10(0.7), 0.1)
        return ml_gas, ml_disk, ml_bulge

    @staticmethod
    def rvs_inclination(mean, std, seed=None):
        """
        Sample a new inclination estimate in degrees. Ensures it is in [30, 90]
        degrees.

        Parameters
        ----------
        mean, std : floats
            The mean and standard deviation of the inclination.
        seed : int, optional
            Random seed.

        Returns
        -------
        inc : float
        """
        gen = numpy.random.RandomState(seed)
        n_attempts = 1000
        for i in range(n_attempts):
            inc = gen.normal(mean, std)
            if 0 <= inc <= 90:
                return inc

        raise RuntimeError("Unable to sample inclination in [30, 90] degrees "
                           "for mean {} and std {}.".format(mean, std))

    @staticmethod
    def Vbar2_from_comps(Vgas, Vdisk, Vbulge, ml_gas, ml_disk, ml_bulge,
                         L36_corr):
        r"""
        Calculate :math:`V_{\rm bar}^2` from velocity components and
        :math:`L_{3.6}` scaling.

        Parameters
        ----------
        Vgas, Vdisk, Vbulge : 1-dimensional arrays
            The gas, disk and bulge rotational velocity components.
        ml_gas, ml_disk, ml_bulge : floats
            The gas, disk and bulge mass-to-light ratios.
        L36_corr : float
            Ratio between resampled value of :math:`L_{3.6}` and its SPARC
            value.

        Returns
        -------
        Vbar : 1-dimensional array
            The squared total baryonic rotational velocity.
        """
        return (ml_gas * Vgas * numpy.abs(Vgas)
                + L36_corr * ml_disk * Vdisk * numpy.abs(Vdisk)
                + L36_corr * ml_bulge * Vbulge * numpy.abs(Vbulge))

    @staticmethod
    def _Vbar2_to_gbar(Vbar, r):
        r"""
        Converts :math:`V_{\rm bar}^2` to :math:`g_{\rm bar}`, taking care of
        the unit conversion. See below for more information.

        Parameters
        ----------
        Vbar2 : 1-dimensional array
            Squared baryonic velocity in :math:`\mathrm{km}^2 / \mathrm{s}^2`.
        r : 1-dimensional array
            Galactocentric distance in :math:`\mathrm{kpc}`.

        Returns
        -------
        gbar  : 1-dimensional array
            Baryonic acceleration in :math:`10^{-10} \mathrm{m} / {s}^2`.
        """
        gbar = Vbar * 1e6                  # Convert Vbar from km/s to m /s
        gbar /= r * units.kpc.to(units.m)  # Divide by r in meters -> gbar
        gbar *= 1e10                        # Take out the powers of 10
        return gbar                         # It's actually gbar now...


class MockRAR(BaseMock):
    """
    Mock data generator that follows an analytic RAR. Resamples `Vbar`, `dist`
    and `inc` from their distributions and scatters them by their statistical
    uncertainties.

    For changing the M/L distribution see `self.rvs_mass_to_light()`.
    """
    def rvs_single(self, model, thetas, Vgas, Vdisk, Vbulge, mean_r,
                   mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36,
                   mean_SBdisk, mean_SBbul, mean_Vobs, e_Vobs,
                   mean_logeN=None, e_logeN=None, alpha=None, beta=None,
                   return_log=False, seed=None):
        r"""
        Resample baryonic and observed centripetal acceleration of a single
        galaxy. All parameters must be specific to a single galaxy. Samples
        distance, inclination and mass-to-light ratios from their priors and
        adds statistical uncertainty due to `L36` and `Vobs`. In case of the
        EFE function resamples its EFE value as well.

        If `alpha` and `beta` are supplied rescales the mock `gobs` to be
        :math:`V^{\alpha} / r^{\beta}`.

        Parameters
        ----------
        model : :py:class`Base1D`
            Analytic RAR model.
        thetas : 1-dimensional array
            Parameters of the RAR model.
        Vgas, Vdisk, Vbulge : 1-dimensional arrays
            The gas, disk and bulge rotational velocity components.
        mean_r : 1-dimensional array
            Estimated radial sampling distance.
        mean_dist, e_dist : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy distance. If provided as an
            array all its elements must be equal.
        mean_inc, e_inc : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy inclination. Assumed to be in
            degrees. If provided as an array all its elements must be equal.
        mean_L36, e_L36 : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy luminosity. If provided as an
            array all its elements must be equal.
        mean_SBdisk, mean_SBbul : 1-dimensional arrays
            Disk and bulge surface brigtness.
        mean_Vobs, e_Vobs : 1-dimensional array
            Mean and uncertainty on the observed rotational velocity.
        mean_logeN, e_logeN : floats or 1-dimensional arrays, optional
            Mean and uncertainty of the EFE.
        alpha, beta : floats
            Exponents to rescale `gobs` to :math:`V^{\alpha} / r^{\beta}`,
            optional.
        return_log : bool, optional
            Whether to return logarithms. By default `False`.
        seed : int, optional
            Random seed.

        Returns
        -------
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y: 1-dimensional arrays
            The baryonic rotational velocity, baryonic centripetal
            acceleration, luminosity, disk surface brightness, bulge surface
            brightness, total surface brightness, galactocentric radial
            distance and observed centripetal acceleration
            or :math:`V^{\alpha} / r^{\beta}`.
        """
        if model.name == "EFE" and (mean_logeN is None or e_logeN is None):
            raise ValueError("For the `EFE` model must give the `eN` pars.")

        # Horrible, horrible style. RC consistency of global properties
        mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_logeN, e_logeN = _check_globprob(mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_logeN, e_logeN)  # noqa
        # Sample the EFE parameter and rewrite it
        if model.name == "EFE":
            gen = numpy.random.RandomState(seed)
            eN_rvs = gen.normal(mean_logeN, e_logeN)
            thetas[1] = eN_rvs

        # Sample priors
        rvs_mls = self.rvs_mass_to_light(seed)                  # M/L ratios
        rvs_L36 = rvs_positive(1., e_L36 / mean_L36, seed)      # L36
        rvs_dist = rvs_positive(mean_dist, e_dist)              # Distance
        rvs_inc = self.rvs_inclination(mean_inc, e_inc, seed)   # Inclination

        # Calculate gbar, independent of distance, scatter by L36
        Vbar2 = self.Vbar2_from_comps(Vgas, Vdisk, Vbulge, *rvs_mls, rvs_L36)
        gbar = self._Vbar2_to_gbar(Vbar2, mean_r)

        # gbar -> gobs, distance, inclination and Vobs stat err corrections
        y = 10**model.predict(numpy.log10(gbar), thetas)

        y *= rvs_dist / mean_dist
        y *= (numpy.sin(numpy.deg2rad(rvs_inc))
              / numpy.sin(numpy.deg2rad(mean_inc)))**2
        y *= rvs_positive(1., 2 * e_Vobs / mean_Vobs, seed)

        # Remaining derived quantities. For Vbar only must consider the
        # distance dependence.
        Vbar = Vbar2**0.5
        Vbar *= numpy.sqrt(rvs_dist / mean_dist)
        L36 = mean_L36 * rvs_L36
        SBdisk = self.copy_replace_zeros(mean_SBdisk, 0.001) * rvs_L36
        SBbul = self.copy_replace_zeros(mean_SBbul, 0.001) * rvs_L36
        SB = SBdisk + SBbul
        r = mean_r * (rvs_dist / mean_dist)

        # Optinally rescale for fitting V^\alpha / r^\beta
        if alpha is not None or beta is not None:
            alpha = 2 if alpha is None else alpha
            beta = 1 if beta is None else beta

            # Convert gobs to Vobs first in km/s
            y = numpy.sqrt(y * r) * (1e-13 * KPC2KM)**0.5
            # Now get the right powers of alpha and beta
            y = y**alpha / r**beta

        if return_log:
            return (numpy.log10(Vbar), numpy.log10(gbar), numpy.log10(L36),
                    numpy.log10(SBdisk), numpy.log10(SBbul), numpy.log10(SB),
                    numpy.log10(r), numpy.log10(y))
        return Vbar, gbar, L36, SBdisk, SBbul, SB, r, y

    def rvs(self, model, thetas, index, Vgas, Vdisk, Vbulge, mean_r,
            mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_SBdisk,
            mean_SBbul, mean_Vobs, e_Vobs, mean_logeN=None, e_logeN=None,
            alpha=None, beta=None, return_log=False, seed=None):
        r"""
        Resample baryonic and observed centripetal acceleration of a many
        galaxies, alls `self.rvs_single(...)` on every galaxy.

        If `alpha` and `beta` are specified then the second returned value is
        not `gobs`. See the function referenced above for more information.

        Parameters
        ----------
        model : :py:class`Base1D`
            Analytic RAR model.
        thetas : 1-dimensional array
            Parameters of the RAR model.
        index : 1-dimensional array
            Indices to distinguish individual galaxies from the passed in
            arrays.
        Vgas, Vdisk, Vbulge : 1-dimensional arrays
            The gas, disk and bulge rotational velocity components.
        mean_r : 1-dimensional array
            Estimated radial sampling distance.
        mean_dist, e_dist : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy distance. If provided as an
            array all its elements must be equal.
        mean_inc, e_inc : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy inclination. Assumed to be in
            degrees. If provided as an array all its elements must be equal.
        mean_L36, e_L36 : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy luminosity. If provided as an
            array all its elements must be equal.
        mean_SBdisk, mean_SBbul : 1-dimensional arrays
            Disk and bulge surface brigtness.
        mean_Vobs, e_Vobs : 1-dimensional array
            Mean and uncertainty on the observed rotational velocity.
        mean_logeN, e_logeN : floats or 1-dimensional arrays, optional
            Mean and uncertainty of the EFE.
        alpha, beta : floats
            Exponents to rescale `gobs` to :math:`V^{\alpha} / r^{\beta}`,
            optional.
        return_log : bool, optional
            Whether to return logarithms. By default `False`.
        seed : int, optional
            Random seed.

        Returns
        -------
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y: 1-dimensional arrays
            The baryonic rotational velocity, baryonic centripetal
            acceleration, luminosity, disk surface brightness, bulge surface
            brightness, total surface brightness, galactocentric radial
            distance and observed centripetal acceleration
            or :math:`V^{\alpha} / r^{\beta}`.
        """
        unique_index = numpy.unique(index)
        # Spread out the seeds
        gen = numpy.random.RandomState(seed)
        # 1e6 should be large enough!
        seeds = gen.choice(numpy.arange(1e6), unique_index.size, replace=False)
        seeds = seeds.astype(int)

        # Preallocate output arrays
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y = [
            numpy.full_like(mean_Vobs, numpy.nan) for __ in range(8)]

        for i, ind in enumerate(unique_index):
            m = ind == index
            mean_logeN_i = None if mean_logeN is None else mean_logeN[m]
            e_logeN_i = None if e_logeN is None else e_logeN[m]

            Vbar[m], gbar[m], L36[m], SBdisk[m], SBbul[m], SB[m], r[m], y[m] = self.rvs_single(  # noqa
                model, thetas, Vgas[m], Vdisk[m], Vbulge[m], mean_r[m],
                mean_dist[m], e_dist[m], mean_inc[m], e_inc[m], mean_L36[m],
                e_L36[m], mean_SBdisk[m], mean_SBbul[m],
                mean_Vobs[m], e_Vobs[m], mean_logeN_i, e_logeN_i,
                alpha, beta, return_log=return_log, seed=seeds[i])

        return Vbar, gbar, L36, SBdisk, SBbul, SB, r, y


class MockSBJobs(BaseMock):
    r"""
    Mock data generator that follows a relation between
    :math:`\Sigma_{\rm tot}` and :math:`J_{\rm obs}`. Resamples `Vbar`, `dist`
    and `inc` from their distributions and scatters them by their statistical
    uncertainties.

    For changing the M/L distribution see `self.rvs_mass_to_light()`.
    """
    def rvs_single(self, model, thetas, Vgas, Vdisk, Vbulge, mean_r,
                   mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36,
                   mean_SBdisk, mean_SBbul, mean_Vobs, e_Vobs,
                   mean_logeN=None, e_logeN=None, alpha=None, beta=None,
                   return_log=False, seed=None):
        r"""
        Resample baryonic and dynamical properties of a galaxy. All parameters
        must be specific to a single galaxy. Samples distance, inclination
        and mass-to-light ratios from their priors and adds statistical
        uncertainty due to `L36` and `Vobs`.

        If `alpha` and `beta` are supplied rescales the mock `gobs` to be
        :math:`V^{\alpha} / r^{\beta}`.

        Parameters
        ----------
        model : :py:class`Base1D`
            Analytic model connecting :math:`\Sigma_{\rm tot}` and
            :math:`J_{\rm obs}`.
        thetas : 1-dimensional array
            Parameters of the model.
        Vgas, Vdisk, Vbulge : 1-dimensional arrays
            The gas, disk and bulge rotational velocity components.
        mean_r : 1-dimensional array
            Estimated radial sampling distance.
        mean_dist, e_dist : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy distance. If provided as an
            array all its elements must be equal.
        mean_inc, e_inc : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy inclination. Assumed to be in
            degrees. If provided as an array all its elements must be equal.
        mean_L36, e_L36 : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy luminosity. If provided as an
            array all its elements must be equal.
        mean_SBdisk, mean_SBbul : 1-dimensional arrays
            Disk and bulge surface brigtness.
        mean_Vobs, e_Vobs : 1-dimensional array
            Mean and uncertainty on the observed rotational velocity.
        mean_logeN, e_logeN : floats or 1-dimensional arrays, optional
            Mean and uncertainty of the EFE.
        alpha, beta : floats
            Exponents to rescale `gobs` to :math:`V^{\alpha} / r^{\beta}`,
            optional.
        return_log : bool, optional
            Whether to return logarithms. By default `False`.
        seed : int, optional
            Random seed.

        Returns
        -------
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y: 1-dimensional arrays
            The baryonic rotational velocity, baryonic centripetal
            acceleration, luminosity, disk surface brightness, bulge surface
            brightness, total surface brightness, galactocentric radial
            distance and observed centripetal acceleration
            or :math:`V^{\alpha} / r^{\beta}`.
        """
        # Horrible, horrible style. RC consistency of global properties
        mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_logeN, e_logeN = _check_globprob(mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_logeN, e_logeN)  # noqa

        # Sample priors
        rvs_mls = self.rvs_mass_to_light(seed)                  # M/L ratios
        rvs_L36 = rvs_positive(1., e_L36 / mean_L36, seed)      # L36
        rvs_dist = rvs_positive(mean_dist, e_dist)              # Distance
        rvs_inc = self.rvs_inclination(mean_inc, e_inc, seed)   # Inclination

        # Calculate SB, independent of distance, scatter by L36
        SBdisk = self.copy_replace_zeros(mean_SBdisk, 0.001) * rvs_L36
        SBbul = self.copy_replace_zeros(mean_SBbul, 0.001) * rvs_L36
        SB = SBdisk + SBbul

        # From this calculate Jobs. distance, inclination and Vobs stat err
        Jobs = 10**model.predict(numpy.log10(SB), thetas)
        Jobs *= (rvs_dist / mean_dist)**2
        Jobs *= (numpy.sin(numpy.deg2rad(rvs_inc))
                 / numpy.sin(numpy.deg2rad(mean_inc)))**3
        Jobs *= rvs_positive(1., 3 * e_Vobs / mean_Vobs, seed)

        # Calculate the remaining derived quantities. For Vbar only must
        # consider the distance dependence.
        Vbar2 = self.Vbar2_from_comps(Vgas, Vdisk, Vbulge, *rvs_mls, rvs_L36)
        gbar = self._Vbar2_to_gbar(Vbar2, mean_r)
        Vbar = Vbar2**0.5
        Vbar *= numpy.sqrt(rvs_dist / mean_dist)
        L36 = mean_L36 * rvs_L36
        r = mean_r * (rvs_dist / mean_dist)

        y = Jobs

        # Optionally rescale for fitting V^\alpha / r^\beta
        if alpha is not None or beta is not None:
            alpha = 2 if alpha is None else alpha
            beta = 1 if beta is None else beta

            # Convert Jobs to Vobs in km/s
            y = (r**2 * Jobs)**(1. / 3)
            # Now get the right powers of alpha and beta
            y = y**alpha / r**beta

        if return_log:
            return (numpy.log10(Vbar), numpy.log10(gbar), numpy.log10(L36),
                    numpy.log10(SBdisk), numpy.log10(SBbul), numpy.log10(SB),
                    numpy.log10(r), numpy.log10(y))
        return Vbar, gbar, L36, SBdisk, SBbul, SB, r, y

    def rvs(self, model, thetas, index, Vgas, Vdisk, Vbulge, mean_r,
            mean_dist, e_dist, mean_inc, e_inc, mean_L36, e_L36, mean_SBdisk,
            mean_SBbul, mean_Vobs, e_Vobs, mean_logeN=None, e_logeN=None,
            alpha=None, beta=None, return_log=False, seed=None):
        r"""
        Resample baryonic and dynamical properties of galaxies. Calls
        `self.rvs_single(...)` on every galaxy.

        If `alpha` and `beta` are specified then the second returned value is
        not `gobs`. See the function referenced above for more information.

        Parameters
        ----------
        model : :py:class`Base1D`
            Analytic RAR model.
        thetas : 1-dimensional array
            Parameters of the RAR model.
        index : 1-dimensional array
            Indices to distinguish individual galaxies from the passed in
            arrays.
        Vgas, Vdisk, Vbulge : 1-dimensional arrays
            The gas, disk and bulge rotational velocity components.
        mean_r : 1-dimensional array
            Estimated radial sampling distance.
        mean_dist, e_dist : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy distance. If provided as an
            array all its elements must be equal.
        mean_inc, e_inc : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy inclination. Assumed to be in
            degrees. If provided as an array all its elements must be equal.
        mean_L36, e_L36 : floats or 1-dimensional arrays
            Mean and uncertainty on the galaxy luminosity. If provided as an
            array all its elements must be equal.
        mean_SBdisk, mean_SBbul : 1-dimensional arrays
            Disk and bulge surface brigtness.
        mean_Vobs, e_Vobs : 1-dimensional array
            Mean and uncertainty on the observed rotational velocity.
        mean_logeN, e_logeN : floats or 1-dimensional arrays, optional
            Mean and uncertainty of the EFE.
        alpha, beta : floats
            Exponents to rescale `gobs` to :math:`V^{\alpha} / r^{\beta}`,
            optional.
        return_log : bool, optional
            Whether to return logarithms. By default `False`.
        seed : int, optional
            Random seed.

        Returns
        -------
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y: 1-dimensional arrays
            The baryonic rotational velocity, baryonic centripetal
            acceleration, luminosity, disk surface brightness, bulge surface
            brightness, total surface brightness, galactocentric radial
            distance and observed centripetal acceleration
            or :math:`V^{\alpha} / r^{\beta}`.
        """
        unique_index = numpy.unique(index)
        # Spread out the seeds
        gen = numpy.random.RandomState(seed)
        # 1e6 should be large enough!
        seeds = gen.choice(numpy.arange(1e6), unique_index.size, replace=False)
        seeds = seeds.astype(int)

        # Preallocate output arrays
        Vbar, gbar, L36, SBdisk, SBbul, SB, r, y = [
            numpy.full_like(mean_Vobs, numpy.nan) for __ in range(8)]

        for i, ind in enumerate(unique_index):
            m = ind == index
            mean_logeN_i = None if mean_logeN is None else mean_logeN[m]
            e_logeN_i = None if e_logeN is None else e_logeN[m]

            Vbar[m], gbar[m], L36[m], SBdisk[m], SBbul[m], SB[m], r[m], y[m] = self.rvs_single(  # noqa
                model, thetas, Vgas[m], Vdisk[m], Vbulge[m], mean_r[m],
                mean_dist[m], e_dist[m], mean_inc[m], e_inc[m], mean_L36[m],
                e_L36[m], mean_SBdisk[m], mean_SBbul[m],
                mean_Vobs[m], e_Vobs[m], mean_logeN_i, e_logeN_i,
                alpha, beta, return_log=return_log, seed=seeds[i])

        return Vbar, gbar, L36, SBdisk, SBbul, SB, r, y


class ToyExponentialModel:
    r"""
    Toy model to predict :math:`g_{\rm bar}` for an exponential disk.
    """
    def mtot(self, Rdisk, SBdisk0):
        r"""
        Total mass of the disk from an exponential profile model. Assumes
        mass-to-light ratio of 1.

        Parameters
        ----------
        Rdisk : float
            Disk scale length in :math:`\mathrm{kpc}`.
        SBdisk0 : float
            Disk central surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.

        Returns
        -------
        M : float
            Total mass of the disk in :math:`\mathrm{M}_{\odot}`.
        """
        return 2 * numpy.pi * (1e3 * Rdisk)**2 * SBdisk0

    def surface_brightness(self, r, Rdisk, SBdisk0):
        r"""
        Surface brightness of the disk from an exponential profile model.

        Parameters
        ----------
        r : float
            Galactocentric radius in :math:`\mathrm{kpc}`.
        Rdisk : float
            Disk scale length in :math:`\mathrm{kpc}`.
        SBdisk0 : float
            Disk central surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.

        Returns
        -------
        SB : float
            Disk surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.
        """
        return SBdisk0 * numpy.exp(-r / Rdisk)

    def gbar_spherical(self, r, Rdisk, SBdisk0):
        r"""
        Predict :math:`g_{\rm bar}` from an exponential profile model but
        assuming spherical symmetry. Assumes mass-to-light ratio of 1.

        Parameters
        ----------
        r : float
            Galactocentric radius in :math:`\mathrm{kpc}`.
        Rdisk : float
            Disk scale length in :math:`\mathrm{kpc}`.
        SBdisk0 : float
            Disk central surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.

        Returns
        -------
        gbar : float
            Baryonic centripetal acceleration in units of
            :math:`10^{-10} \mathrm{m} / \mathrm{s}^2`.`:
        """
        G = 6.6743e-11   # m^3 kg^-1 s^-2
        Msun = 1.989e30  # kg
        r_meters = r * units.kpc.to(units.m)

        mtot = self.mtot(Rdisk, SBdisk0) * Msun

        gbar = G * mtot / r_meters**2
        gbar *= 1 - (1 + r / Rdisk) * numpy.exp(-r / Rdisk)
        gbar *= 1e10
        return gbar

    def __call__(self, r, Rdisk, SBdisk0):
        r"""
        Predict :math:`g_{\rm bar}` from an exponential profile model. Assumes
        mass-to-light ratio of 1.

        Parameters
        ----------
        r : float
            Galactocentric radius in :math:`\mathrm{kpc}`.
        Rdisk : float
            Disk scale length in :math:`\mathrm{kpc}`.
        SBdisk0 : float
            Disk central surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.

        Returns
        -------
        Baryonic centripetal acceleration in units of
            :math:`10^{-10} \mathrm{m} / \mathrm{s}^2`.`:
        """
        G = 6.6743e-11   # m^3 kg^-1 s^-2
        Msun = 1.989e30  # kg
        SBdisk0 = numpy.copy(SBdisk0)
        SBdisk0 *= Msun / (units.pc.to(units.m))**2
        y = r / (2 * Rdisk)
        out = i0(y) * k0(y) - i1(y) * k1(y)
        out *= 2 * numpy.pi * G * SBdisk0
        out *= 1e10
        return out


class ToyKuzminModel:
    r"""
    Toy model to predict :math:`g_{\rm bar}` for a Kuzmin potential.
    """
    def surface_brightness(self, r, mtot, a):
        r"""
        Surface brightness of the disk from a Kuzmin model.

        Parameters
        ----------
        r : float
            Galactocentric radius in :math:`\mathrm{kpc}`.
        mtot : float
            Total mass of the disk in :math:`\mathrm{M}_{\odot}`.
        a : float
            Kuzmin potential scale length in :math:`\mathrm{kpc}`.

        Returns
        -------
        SB : float
            Disk surface brightness in
            :math:`\mathrm{L}_{\odot} / \mathrm{pc}^2`.
        """
        return a * mtot / (2 * numpy.pi * (r**2 + a**2)**(3 / 2)) * 1e-6

    def __call__(self, r, mtot, a):
        r"""
        Predict :math:`g_{\rm bar}` for a Kuzmin potential. Assumes
        mass-to-light ratio of 1.

        Parameters
        ----------
        r : float
            Galactocentric radius in :math:`\mathrm{kpc}`.
        mtot : float
            Total mass of the disk in :math:`\mathrm{M}_{\odot}`.
        a : float
            Kuzmin potential scale length in :math:`\mathrm{kpc}`.

        Returns
        -------
        Baryonic centripetal acceleration in units of
            :math:`10^{-10} \mathrm{m} / \mathrm{s}^2`.`:
        """
        G = 6.6743e-11   # m^3 kg^-1 s^-2
        mtot *= 1.989e30  # kg
        return G * mtot * r / (r**2 + a**2)**(3 / 2) * 1e10 / units.kpc.to(units.m)**2  # noqa
