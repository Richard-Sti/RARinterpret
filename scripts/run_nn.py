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
MPI script to fit the NN on many splits while keeping hyperparameters fixed.
"""
import warnings
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from os import remove
from os.path import join

import joblib
import numpy
from mpi4py import MPI
from sklearn.impute import SimpleImputer
from taskmaster import master_process, worker_process

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
verbose = nproc == 1

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--features", type=str)
parser.add_argument("--target", type=str, choices=["Vobs", "gobs"])
parser.add_argument("--n_splits", type=int)
parser.add_argument("--test_size", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--patience", type=int)
parser.add_argument("--width", type=int)
parser.add_argument("--dropout_rate", type=float)
parser.add_argument("--add_rarif", type=lambda x: bool(strtobool(x)))
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
features = RARinterpret.parse_features(args.features)

# Set up paths
dumpdir = "/mnt/extraspace/rstiskalek/rar"
kind = "PINN" if args.add_rarif else "NN"
ftemp = join(dumpdir, "temp", "fit_{}_{}_{}_{}".format(
    kind, args.target, args.features, args.test_size) + "_{}.p")
fperm = join("../results/fit", "{}_{}_{}_{}.p".format(
    kind, args.target, args.features, args.test_size))

# Load up all dat
frame = RARinterpret.RARFrame()
test_masks = RARinterpret.make_test_masks(frame["index"], args.n_splits,
                                          test_size=args.test_size,
                                          random_state=args.seed)
X_, y, features = frame.make_Xy(target=args.target, features=features,
                                append_variances=True, dtype=numpy.float32)


def task(n):
    train, test = RARinterpret.train_test_from_mask(test_masks[n, :])
    if test.size == 0:
        test = train

    # Run the imputer. Train only on train and apply everywhere.
    X = numpy.copy(X_)
    imputer = SimpleImputer()
    imputer.fit(X[train])  # Fit only on train
    X = imputer.transform(X)

    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.005, first_decay_steps=500, alpha=1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=schedule, amsgrad=True)
    reg = RARinterpret.PLModel.from_hyperparams(
        X[train], layers=[args.width], opt=opt,
        dropout_rate=args.dropout_rate,)

    hist = reg.train(X[train], y[train], epochs=args.epochs,
                     patience=args.patience, batch_fraction=1/3,
                     val_fraction=1.0, verbose=verbose)

    out = {"loss": reg.score(X[test], y[test]) / test.size * len(frame),
           "ypred": reg.predict(X).numpy().reshape(-1,),
           "epoch_stop": hist.epoch[-1],
           }
    joblib.dump(out, ftemp.format(n))


# Work delegation if-else. rank = 0 delegates the tasks.
if nproc > 1:
    if rank == 0:
        tasks = list(range(args.n_splits))
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(task, comm, verbose=False)
else:
    tasks = list(range(args.n_splits))
    for n in tasks:
        print("{}: completing task `{}`".format(datetime.now(), n), flush=True)
        task(n)


comm.Barrier()
if rank == 0:
    print("Collecting results.", flush=True)
    # Pre-allocate arrays
    loss = numpy.full(args.n_splits, numpy.nan, dtype=numpy.float32)
    epoch_stop = numpy.full(args.n_splits, numpy.nan, dtype=numpy.int32)
    ypred = numpy.full((args.n_splits, len(frame)), numpy.nan,
                       dtype=numpy.float32)

    for n in range(args.n_splits):
        finp = joblib.load(ftemp.format(n))
        loss[n, ...] = finp["loss"]
        ypred[n, ...] = finp["ypred"]
        epoch_stop[n, ...] = finp["epoch_stop"]
        # Remove the temporary file
        remove(ftemp.format(n))

    # Dump it
    print("Saving to `{}`...".format(fperm), flush=True)
    out = {"loss": loss, "ypred": ypred, "epoch_stop": epoch_stop, }
    joblib.dump(out, fperm)
    print("All finished!", flush=True)

# Forcefully quit
comm.Barrier()
quit()
