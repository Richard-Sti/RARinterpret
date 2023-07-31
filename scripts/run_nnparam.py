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
MPI script to fit the NN on many splits while keeping hyperparameters fixed.
"""
import warnings
from argparse import ArgumentParser
from os.path import join

import joblib
import numpy
from optuna import create_study
from sklearn.impute import SimpleImputer

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


# Argument parsing
parser = ArgumentParser()
parser.add_argument("--features", type=str)
parser.add_argument("--target", type=str, choices=["Vobs", "gobs"])
parser.add_argument("--n_trials", type=int)
parser.add_argument("--n_splits", type=int)
parser.add_argument("--test_size", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--patience", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
features = RARinterpret.parse_features(args.features)


# Set up paths
dumpdir = "/mnt/extraspace/rstiskalek/rar/nn"
fout = join("../results/hyper", "{}_{}_{}.p".format("NN", args.target,
                                                    args.features))

# Load up all data
frame = RARinterpret.RARFrame()
test_masks = RARinterpret.make_test_masks(frame["index"], args.n_splits,
                                          test_size=args.test_size,
                                          random_state=args.seed)
X_, y, features = frame.make_Xy(target=args.target, features=features,
                                append_variances=True, dtype=numpy.float32)


def reg_from_trial(trial, X):
    width = trial.suggest_int("width", 4, 128)
    dropout_rate = trial.suggest_float("dropout_rate", 0.001, 0.1)

    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.005, first_decay_steps=500, alpha=1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=schedule, amsgrad=True)
    return RARinterpret.PLModel.from_hyperparams(X, layers=[width], opt=opt,
                                                 dropout_rate=dropout_rate,)


def objective(trial):
    loss = 0.

    for n in range(args.n_splits):
        train, test = RARinterpret.train_test_from_mask(test_masks[n, :])

        # Run the imputer. Train only on train and apply everywhere.
        X = numpy.copy(X_)
        imputer = SimpleImputer()
        imputer.fit(X[train])  # Fit only on train
        X = imputer.transform(X)

        # Create the regressor and score it
        reg = reg_from_trial(trial, X[train])
        reg.train(X[train], y[train], epochs=args.epochs,
                  patience=args.patience, batch_fraction=1/3,
                  verbose=False)
        loss += reg.score(X[test], y[test])
    return loss


study = create_study(direction="minimize")
study.optimize(objective, n_trials=args.n_trials)

# Evaluate the average scaled loss
loss = numpy.asanyarray([tr.value for tr in study.trials])
loss *= len(frame) / test_masks.sum()

out = {"trials": study.trials,
       "best_params": study.best_params,
       "loss": loss,
       }
print("Saving to `{}`...".format(fout), flush=True)
joblib.dump(out, fout)
print("All finished!", flush=True)
