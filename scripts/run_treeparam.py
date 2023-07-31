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
Script to run hyperparameter optimisation for tree models over the entire data
sets.
"""
from argparse import ArgumentParser
from distutils.util import strtobool
from os.path import join

import jax
import joblib
import numpy
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret
jax.config.update("jax_platform_name", "cpu")

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--reg", type=str, choices=["ET", "XGB"])
parser.add_argument("--features", type=str)
parser.add_argument("--target", type=str, choices=["Vobs", "gobs"])
parser.add_argument('--add_PCA', type=lambda x: bool(strtobool(x)))
parser.add_argument("--trials", type=int)
parser.add_argument("--nfolds", type=int, default=5)
parser.add_argument("--ncpu", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
features = RARinterpret.parse_features(args.features)

# Set up paths
if args.add_PCA:
    fout = join("../results/hyper", "{}_{}_{}_PCA.p")
else:
    fout = join("../results/hyper", "{}_{}_{}.p")
fout = fout.format(args.reg, args.target, args.features)


###############################################################################
#                  Regressors and hyperparameter distributions                #
###############################################################################

if args.reg == "ET":
    reg = RARinterpret.basic_pipeline(
        ExtraTreesRegressor(n_jobs=1, bootstrap=True), with_PCA=args.add_PCA)
    dist = {
        "estimator__n_estimators": IntDistribution(64, 128),
        "estimator__max_depth": IntDistribution(2, 16),
        "estimator__min_samples_split": IntDistribution(2, 32),
        "estimator__max_features": CategoricalDistribution(["sqrt", "log2",
                                                            None]),
        "estimator__min_impurity_decrease": FloatDistribution(1e-14, 0.5,
                                                              log=True),
        "estimator__ccp_alpha": FloatDistribution(1e-14, 0.5, log=True),
        "estimator__max_samples": FloatDistribution(0.1, 0.99),
        }
elif args.reg == "XGB":
    reg = RARinterpret.basic_pipeline(
        XGBRegressor(n_jobs=1), with_PCA=args.add_PCA)
    dist = {
        "estimator__n_estimators": IntDistribution(16, 128),
        "estimator__max_depth": IntDistribution(2, 8),
        "estimator__booster": CategoricalDistribution(["gbtree", "dart"]),
        "estimator__learning_rate": FloatDistribution(0.01, 0.99),
        "estimator__gamma": FloatDistribution(0, 10),
        "estimator__min_child_weight": FloatDistribution(0.5, 2.5),
        "estimator__subsample": FloatDistribution(0.5, 1),
        }
else:
    raise NotImplementedError("Not implemented yet.")

if args.add_PCA and len(features) > 1:
    dist.update({"PCA__n_components": IntDistribution(1, len(features))})

###############################################################################
#                               Analysis                                      #
###############################################################################


# Load data
frame = RARinterpret.RARFrame()
X, y, features = frame.make_Xy(target=args.target, features=features)
# Approximate sample weights if not a NN
if args.target == "gobs" and features[0] == "gbar":
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

search = OptunaSearchCV(reg, dist, cv=GroupKFold(n_splits=args.nfolds),
                        scoring="neg_mean_squared_error",
                        random_state=args.seed, n_jobs=-1,
                        n_trials=args.trials, verbose=False)
search.fit(X, y, groups=frame["index"], estimator__sample_weight=sample_weight)

out = {"trials": search.trials_, "best_params": search.best_params_}
print("Dumping results to `{}`.".format(fout), flush=True)
joblib.dump(out, fout)
print("All finished!", flush=True)
