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
"""ML modelling functions."""
import numpy
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

###############################################################################
#                           Train-test splitting                              #
###############################################################################


def make_test_masks(groups, n_splits, test_size=0.2, random_state=42):
    """
    Make a `n_splits` test masks with a group shuffle split. If `test_size` is
    0 then `test_mask` is `False` everywhere.

    Parameters
    ----------
    groups : 1-dimensional array
        Group indices (galaxy indices)
    n_splits : int
        Number of splits.
    test_size : float, optional
        Fractional test size. By default 0.2
    random_state : int, optional
        Random seed. By default 42.

    Returns
    -------
    test_mask : 2-dimensional array
        Boolean array of shape `(n_splits, n_samples)`, where `True` indicates
        that the sample belongs to the test set.
    """
    if test_size == 0.:
        return numpy.zeros((n_splits, groups.size), dtype=bool)

    cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                           random_state=random_state)
    test_mask = numpy.zeros((n_splits, groups.size), dtype=bool)
    Xfid = groups.reshape(-1, 1)
    for i, (__, test) in enumerate(cv.split(Xfid, groups=groups)):
        test_mask[i, test] = True

    return test_mask


def train_test_from_mask(test_mask):
    """
    Return train and test indices from a test mask.

    Parameters
    ----------
    test_mask : 1-dimensional array
        Boolean array of shape `(n_samples, )`, where `True` indicates that the
        sample belongs to the test set.

    Returns
    -------
    train : 1-dimensional array
        Training set indices.
    test : 1-dimensional array
        Test set indices.
    """
    if test_mask.ndim != 1:
        raise TypeError("`test_mask` must be a 1-dimensional array.")
    _x = numpy.arange(test_mask.size)
    return _x[~test_mask], _x[test_mask]


def split_jobs(Njobs, Ncpu):
    """
    Split `Njobs` amongst `Ncpu`.

    Parameters
    ----------
    Njobs : int
        Number of jobs.
    Ncpu : int
        Number of CPUs.

    Returns
    -------
    jobs : list of lists of integers
        Outer list of each CPU and inner lists for CPU's jobs.
    """
    njobs_per_cpu, njobs_remainder = divmod(Njobs, Ncpu)
    jobs = numpy.arange(njobs_per_cpu * Ncpu).reshape((njobs_per_cpu, Ncpu)).T

    jobs = jobs.tolist()
    for i in range(njobs_remainder):
        jobs[i].append(njobs_per_cpu * Ncpu + i)

    return jobs


###############################################################################
#                               ML fitting                                    #
###############################################################################


def basic_pipeline(estimator, with_PCA=False, scaler=StandardScaler()):
    """
    Get a pipeline consisting of `imputer`, `scaler` and `estimator` steps.
    Optionally puts `PCA` between `scaler` and `estimator`.

    Parameters
    -----------
    estimator : sklearn estimator instance
        An unfitted estimator.
    with_PCA : bool, optional
        Whether to add a PCA step. By default `False`.
    scaler : sklearn scaler instance, optional
        An appropriate scaler. By default standard scaling.

    Returns
    -------
    pipeline : :py:class:`sklearn.pipeline.Pipeline`
        An unfitted sklearn pipeline.
    """
    if with_PCA:
        steps = [("imputer", SimpleImputer()),
                 ("scaler", scaler),
                 ("PCA", PCA()),
                 ("estimator", estimator)
                 ]
    else:
        steps = [("imputer", SimpleImputer()),
                 ("scaler", scaler),
                 ("estimator", estimator)
                 ]
    return Pipeline(steps)


def est_fit_score(est, X, y, test_mask, sample_weight, n_repeats=100):
    """
    Shortcut to fit and score an estimator on the test samples. If there are no
    test samples, then the model is instead scored on the training data.

    Parameters
    ----------
    estimator : sklearn estimator instance
        An unfitted estimator.
    X : 2-dimensional array
        The feature array of shape (n_samples, n_features).
    y : 1- or 2-dimensional array.
        The corresponding target array.
    test_mask : 1-dimensional array
        Boolean array of shape `(n_samples, )`, where `True` indicates that the
        sample belongs to the test set.
    sample_weight : 1-dimensional array
        The sample weights. By default `None` and used only if the
        estimator accepts `sample_weight` argument.
    n_repeats : int, optional
        Number of times to permute a feature to calculate the permutation
        importance. By default 100.

    Returns
    -------
    estimator : sklearn estimator instance
        A fitted estimator.
    loss : float
        Test set loss.
    importances : 2-dimensional array of shape `(n_features, 3)`
        Columns are feature importance, mean permutation importance and
        standard deviation of permutation importance, respectively.
    """
    est = clone(est)
    # Unpack parameters
    train, test = train_test_from_mask(test_mask)
    if numpy.sum(test_mask) == 0:
        test = train

    train_w = sample_weight[train] if sample_weight is not None else None
    test_w = sample_weight[test] if sample_weight is not None else None
    # Fit, score
    est.fit(X[train], y[train], estimator__sample_weight=train_w)
    loss = 0.5 * numpy.sum(test_w * (y[test] - est.predict(X[test]))**2)
    # Feature and permutation importance
    if "PCA" not in est.named_steps.keys():
        importance = get_importance(est, X[test, :], y[test],
                                    sample_weight[test], n_jobs=1,
                                    n_repeats=n_repeats)
    else:
        importance = numpy.nan
    return est, loss, importance


def get_importance(pipeline, X, y, weights=None, n_jobs=1, n_repeats=25):
    """
    Get the feature and permutation importance of a fitted pipeline.

    Parameters
    ----------
    pipeline : :py:class:`sklearn.pipeline.Pipeline`
        A fitted sklearn pipeline.
    X : 2-dimensional array
        The input samples of shape (n_samples, n_features).
    y : 1-dimensional array
        The target values of shape (n_samples, ).
    weights : 1-dimensional array, optional
        The target weights. By default `None`.
    n_jobs : int, optional
        Number of jobs to run in parallel for permutation importance. By
        default 1.
    n_repeats : int, optional
        Number of times to permute a feature. By default 25.

    Returns
    -------
    importances : 2-dimensional array of shape `(n_features, 3)`
        Columns are feature importance, mean permutation importance and
        standard deviation of permutation importance, respectively.
    """
    # Pre-allocate array, extract feature importance
    out = numpy.full((X.shape[1], 3), numpy.nan)
    try:
        out[:, 0] = pipeline.named_steps["estimator"].feature_importances_
    except AttributeError:
        pass

    # Calculate permutation importance
    perm = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                  n_jobs=n_jobs, sample_weight=weights,
                                  scoring="neg_mean_squared_error")
    out[:, 1] = perm["importances_mean"]
    out[:, 2] = perm["importances_std"]
    return out
