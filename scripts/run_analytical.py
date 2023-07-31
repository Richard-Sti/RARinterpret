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
Script to fit an analytical function to the RAR.
"""
from argparse import ArgumentParser
from copy import deepcopy
from distutils.util import strtobool

import joblib
import numpy
from scipy.optimize import minimize
from tqdm import trange

# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

# Grab the model we're fitting
parser = ArgumentParser()
parser.add_argument("--model", type=str, choices=["IF", "EFE", "ESR"])
parser.add_argument("--test_size", type=float)
parser.add_argument("--dydx", type=str, choices=["IF", "AD", "No"])
parser.add_argument("--nsplits", type=int)
parser.add_argument("--skip_mocks", type=lambda x: bool(strtobool(x)))
parser.add_argument("--seed", type=int, default=42)
pargs = parser.parse_args()


# Load data
frame = RARinterpret.RARFrame()
x, y = numpy.log10(frame["gbar"]), numpy.log10(frame["gobs"])
varx = frame.generate_log_variance("gbar")
vary = frame.generate_log_variance("gobs")
test_masks = RARinterpret.make_test_masks(
    frame["index"], pargs.nsplits, test_size=pargs.test_size,
    random_state=pargs.seed)

# Load the model
if pargs.model == "IF":
    model = RARinterpret.RARIF()
if pargs.model == "ESR":
    model = RARinterpret.BestRARESR()
else:
    model = RARinterpret.SimpleIFEFE()

# Get the (approximate) gradient
if pargs.dydx == "IF":
    gradmodel = RARinterpret.RARFitInterpolator()
    dydx = gradmodel.dpredict(x, gradmodel.x0)
elif pargs.dydx == "No":
    dydx = numpy.zeros_like(varx)
else:
    dydx = None


def fit(xs, ys, train, test, results):
    """
    Shortcut to fit the model, calculate summary statistics and write output
    to `results`.
    """
    args = (xs[train], ys[train], varx[train], vary[train],)
    if dydx is not None:
        args += (dydx[train],)
    x0 = model.x0 * numpy.random.uniform(0.9, 1.1)
    res = minimize(model, numpy.copy(x0), args=args, method="Nelder-Mead")

    results["thetas"][n, :] = res.x
    args = (res.x, xs[test], ys[test], varx[test], vary[test],)
    if dydx is not None:
        args += (dydx[test],)
    results["loss"][n] = model(*args) / test.size
    results["ypred"][n, :] = model.predict(xs, res.x)


# Pre-allocate output
results_data = {
    "thetas": numpy.full((pargs.nsplits, model.nparams), numpy.nan),
    "loss": numpy.full(pargs.nsplits, numpy.nan),
    "ypred": numpy.full((pargs.nsplits, len(frame)), numpy.nan),
    }
results_mock = deepcopy(results_data)

print("Running...", flush=True)
for n in trange(pargs.nsplits):
    train, test = RARinterpret.train_test_from_mask(test_masks[n, :])
    if numpy.sum(test_masks[n, :]) == 0:
        test = train
    # Fit SPARC data
    fit(x, y, train, test, results_data)
    # Skip for EFE.. some bugs
    if pargs.skip_mocks or model.name == "EFE":
        continue
    # Fit mock data
    xmock, ymock, __ = frame.make_Xy(target="gobs", features="gbar",
                                     gen_model=model,
                                     thetas=numpy.copy(model.x0), seed=n)
    fit(xmock.reshape(-1,), ymock, train, test, results_mock)


fout = "../results/analytical/{}_{}_{}_{}.p"
fdata = fout.format(pargs.model, "data", pargs.test_size, pargs.dydx)
fmock = fout.format(pargs.model, "mock", pargs.test_size, pargs.dydx)
print("Dumping results to.. `{}`, `{}`.".format(fdata, fmock), flush=True)
joblib.dump(results_data, fdata)
joblib.dump(results_mock, fmock)
