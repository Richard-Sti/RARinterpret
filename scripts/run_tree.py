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
MPI script to fit tree regressors on many test-train splits while keeping
their hyperparameters fixed to previously optimised values.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from os import remove
from os.path import join

import jax
import joblib
import numpy
from mpi4py import MPI
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret
jax.config.update("jax_platform_name", "cpu")

# Get MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--reg", type=str, choices=["ET", "XGB"])
parser.add_argument("--features", type=str)
parser.add_argument("--target", type=str, choices=["Vobs", "gobs"])
parser.add_argument('--add_PCA', type=lambda x: bool(strtobool(x)))
parser.add_argument("--n_splits", type=int)
parser.add_argument("--test_size", type=float)
parser.add_argument("--hyper", type=str,
                    choices=["specific", "RAR", "gobs_all"])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
features = RARinterpret.parse_features(args.features)


# Set up paths
dumpdir = "/mnt/extraspace/rstiskalek/rar"
ftemp = join(dumpdir, "temp", "fit_{}_{}_{}_{}_{}".format(
    args.reg, args.target, args.features, args.test_size, args.add_PCA) + "_{}.p")  # noqa
if args.add_PCA:
    fperm = join("../results/fit", "{}_{}_{}_{}_PCA.p".format(
        args.reg, args.target, args.features, args.test_size))
else:
    fperm = join("../results/fit", "{}_{}_{}_{}.p".format(
        args.reg, args.target, args.features, args.test_size))

if args.hyper == "specific":
    if args.add_PCA:
        fhyper = join("../results/hyper", "{}_{}_{}_PCA.p"
                      .format(args.reg, args.target, args.features))
    else:
        fhyper = join("../results/hyper", "{}_{}_{}.p"
                      .format(args.reg, args.target, args.features))
elif args.hyper == "RAR":
    fhyper = join("../results/hyper", "{}_gobs_gbar.p".format(args.reg))
elif args.hyper == "gobs_all":
    fhyper = join("../results/hyper/",
                  "{}_gobs_gbar,r,SBdisk,SBbul,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust.p".format(args.reg))  # noqa
else:
    raise NotImplementedError("Not implemented yet.")

###############################################################################
#                    Regressor and hyperparameter setting                     #
###############################################################################


if args.reg == "ET":
    reg = RARinterpret.basic_pipeline(
        ExtraTreesRegressor(n_jobs=1, bootstrap=True), with_PCA=args.add_PCA)
elif args.reg == "XGB":
    reg = RARinterpret.basic_pipeline(
        XGBRegressor(n_jobs=1), with_PCA=args.add_PCA)
else:
    raise NotImplementedError("Not implemented yet.")

reg.set_params(**joblib.load(fhyper)["best_params"])


###############################################################################
#                               Analysis                                      #
###############################################################################


# Load data and possibly approximate sample weights if not a NN
frame = RARinterpret.RARFrame()
X, y, features = frame.make_Xy(target=args.target, features=features)
test_masks = RARinterpret.make_test_masks(frame["index"], args.n_splits,
                                          test_size=args.test_size,
                                          random_state=args.seed)
if args.reg != "NN" and args.target == "gobs" and features[0] == "gbar":
    print("Approximating sample weights with the RAR IF.", flush=True)
    gradmodel = RARinterpret.RARIF()
    sample_weight = gradmodel.make_weights(
        gradmodel.x0, numpy.log10(frame["gbar"]),
        frame.generate_log_variance("gbar"),
        frame.generate_log_variance("gobs"))
else:
    print("Approximating sample weights with the target variance only.",
          flush=True)
    sample_weight = 1 / frame.generate_log_variance(args.target)


jobs = RARinterpret.split_jobs(args.n_splits, nproc)[rank]
for i, n in enumerate(jobs):
    print("{}: rank {} completing {} / {} jobs.".format(
        datetime.now(), rank, i + 1, len(jobs)))

    reg, loss, imp = RARinterpret.est_fit_score(reg, X, y, test_masks[n, :],
                                                sample_weight)
    out = {"loss": loss, "imp": imp, "ypred": reg.predict(X)}
    joblib.dump(out, ftemp.format(n))

# Wait for all ranks to finish and then combine results
comm.Barrier()
if rank == 0:
    # Pre-allocate arrays
    loss = numpy.full(args.n_splits, numpy.nan)
    importance = numpy.full((args.n_splits, len(features), 3), numpy.nan)
    ypred = numpy.full((args.n_splits, len(frame)), numpy.nan)

    for n in range(args.n_splits):
        finp = joblib.load(ftemp.format(n))
        loss[n] = finp.get("loss", numpy.nan)
        importance[n, ...] = finp.get("imp", numpy.nan)
        ypred[n, ...] = finp.get("ypred", numpy.nan)
        # Remove the temporary file
        remove(ftemp.format(n))

    if args.test_size > 0:
        loss /= numpy.sum(test_masks, axis=1)

    # Dump it
    print("Saving to `{}`...".format(fperm), flush=True)
    out = {"loss": loss,
           "ypred": ypred,
           "importance": importance,
           "features": features,
           "test_masks": test_masks,
           }
    joblib.dump(out, fperm)
    print("All finished!")
