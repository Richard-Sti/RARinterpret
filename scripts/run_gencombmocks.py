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
"""MPI script to fit a generic combination of Vobs and r."""
from argparse import ArgumentParser
from datetime import datetime
from os import remove
from os.path import join

import jax
import joblib
import numpy
import tensorflow as tf
from mpi4py import MPI
from scipy.stats import kendalltau
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from taskmaster import master_process, worker_process

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

# JAX silence warnings
jax.config.update("jax_platform_name", "cpu")
# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--reg", type=str, choices=["NN", "ET", "kend", "linear"])
parser.add_argument("--features", type=str)
parser.add_argument("--gen_model", type=str, default="RAR",
                    choices=["RAR", "SB-Jobs"])
parser.add_argument("--nresample", type=int)
parser.add_argument("--n_splits", type=int)
parser.add_argument("--test_size", type=float)
parser.add_argument("--n_theta", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--patience", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--dropout_rate", type=float)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
features = RARinterpret.parse_features(args.features)

# Get the grid
thetas = numpy.hstack(
    [numpy.linspace(0, numpy.pi / 2, int(0.75 * args.n_theta)),
     numpy.linspace(numpy.pi / 2, numpy.pi, int(0.25 * args.n_theta))])
grid = numpy.vstack([numpy.cos(thetas), numpy.sin(thetas)]).T
ngrid = grid.shape[0]

# Set up paths
dumpdir = "/mnt/extraspace/rstiskalek/rar"
reg = args.reg
if args.gen_model == "SB-Jobs":
    reg += "SB-Jobs"
ftemp = join(dumpdir, "temp",
             "gencomb_mock_{}_{}".format(reg, args.features) + "_{}.npz")
fperm = join("../results/gencomb",
             "mock_{}_{}.p".format(reg, args.features))

# Load data
frame = RARinterpret.RARFrame()
if args.gen_model == "RAR":
    genmodel = RARinterpret.RARIF()
else:
    genmodel = RARinterpret.SJCubic()
test_masks = RARinterpret.make_test_masks(
    frame["index"], args.n_splits, test_size=args.test_size,
    random_state=args.seed)
nfeatures = len(features)


def do_task(n):
    alpha, beta = grid[n, :]
    # Loop over the splits
    _scores = numpy.full((args.nresample, args.n_splits), numpy.nan)
    _imp = numpy.full((args.nresample, args.n_splits, nfeatures, 3), numpy.nan)
    for i in range(args.nresample):
        _X, y, __ = frame.make_Xy(
            target=(alpha, beta), features=features, gen_model=genmodel,
            thetas=genmodel.x0, mock_kind=args.gen_model,
            append_variances=True if args.reg == "NN" else False)
        for j in range(args.n_splits):
            train, test = RARinterpret.train_test_from_mask(test_masks[j, :])
            if test.size == 0:
                test = train

            # Run the imputer. Train only on train and apply everywhere.
            X = numpy.copy(_X)
            imputer = SimpleImputer()
            imputer.fit(X[train])  # Fit only on train
            X = imputer.transform(X)
            if args.reg == "NN":
                schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=0.005, first_decay_steps=500,
                    alpha=1e-3)
                opt = tf.keras.optimizers.Adam(learning_rate=schedule,
                                               amsgrad=True)
                reg = RARinterpret.PLModel.from_hyperparams(
                    X[train], layers=[args.width], opt=opt,
                    dropout_rate=args.dropout_rate,)

                reg.train(X[train], y[train], epochs=args.epochs,
                          patience=args.patience, batch_fraction=1/3,
                          val_fraction=1.0, verbose=False)
                _scores[i, j] = reg.score(X[test], y[test]) / test.size
            elif args.reg == "ET":
                reg = RARinterpret.basic_pipeline(
                    ExtraTreesRegressor(n_jobs=1, bootstrap=True))
                if nfeatures == 1:
                    fhyper = join("../results/hyper", "ET_gobs_gbar.p")
                else:
                    fhyper = join("../results/hyper",
                                  "ET_gobs_gbar,r,SBdisk,SBbul,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust.p")  # noqa
                reg.set_params(**joblib.load(fhyper)["best_params"])
                weights = 1. / frame.generate_log_variance((alpha, beta))
                reg.fit(X[train], y[train],
                        estimator__sample_weight=weights[train])
                _scores[i, j] = 0.5 * numpy.mean(
                    (reg.predict(X[test]) - y[test])**2 * weights[test])
                _imp[i, j, ...] = RARinterpret.get_importance(
                    reg, X[test], y[test], weights[test], n_repeats=50)
            elif args.reg == "kend":
                assert X.shape[1] == 1, "Only one feature allowed for Kendall."
                _scores[i, ...] = kendalltau(X[train].reshape(-1,), y[train])[0]  # noqa
            else:
                reg = LinearRegression()
                reg.fit(X[train], y[train])
                _scores[i, j] = reg.score(X[test], y[test]) / test.size
    numpy.savez(ftemp.format(n), scores=_scores, imp=_imp)


# Work delegation if-else. rank = 0 delegates the tasks.
if nproc > 1:
    if rank == 0:
        tasks = list(range(ngrid))
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_task, comm, verbose=False)
else:
    tasks = list(range(ngrid))
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task),
              flush=True)
        do_task(task)


comm.Barrier()
if rank == 0:
    scores = numpy.full((ngrid, args.nresample, args.n_splits), numpy.nan)
    imps = numpy.full((ngrid, args.nresample, args.n_splits, nfeatures, 3),
                      numpy.nan)
    for n in range(ngrid):
        f = numpy.load(ftemp.format(n))
        scores[n, ...] = f["scores"]
        imps[n, ...] = f["imp"]
        remove(ftemp.format(n))

    # Dump it
    print("Saving to `{}`...".format(fperm), flush=True)
    out = {"features": features,
           "thetas": thetas,
           "grid": grid,
           "imps": imps,
           "scores": scores,
           "test_masks": test_masks,
           }
    joblib.dump(out, fperm)

    print("All finished!", flush=True)
