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
MPI parallelised partial correlation calculation.
"""
from argparse import ArgumentParser
from datetime import datetime
from os import remove
from os.path import join

import joblib
import numpy
from mpi4py import MPI
from scipy.stats import kendalltau, spearmanr

# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

# Get MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

parser = ArgumentParser()
parser.add_argument("--model", type=str, choices=["IF", "EFE"])
parser.add_argument("--pc_features", type=str)
pargs = parser.parse_args()
features = RARinterpret.parse_features(pargs.pc_features)
if "gbar" in features:
    features.remove("gbar")

# Set up paths
dumpdir = "/mnt/extraspace/rstiskalek/rar"
ftemp = join(dumpdir, "temp", "pc_{}".format(pargs.model) + "_{}.p")
fperm = "../results/pc_{}.p".format(pargs.model)

# Load data
frame = RARinterpret.RARFrame()
Z, __, features = frame.make_Xy(target="gobs", features=features)
log_gbar = numpy.log10(frame["gbar"])
log_gobs = numpy.log10(frame["gobs"])
var_log_gbar = frame.generate_log_variance("gbar")
var_log_gobs = frame.generate_log_variance("gobs")

# Load the model
if pargs.model == "IF":
    model = RARinterpret.RARIF()
else:
    model = RARinterpret.SimpleIFEFE()
gen_model = RARinterpret.SimpleIFEFE()

# Load parameters of the fitted models
fin = "../results/analytical/{}_{}_0.0_AD.p"
fit_data = joblib.load(fin.format(pargs.model, "data"))
fit_mock = joblib.load(fin.format(pargs.model, "mock"))
nsplits = fit_data["thetas"].shape[0]

# for n in trange(nsplits):
jobs = RARinterpret.split_jobs(nsplits, nproc)[rank]
for i, n in enumerate(jobs):
    print("{}: rank {} completing {} / {} jobs.".format(
        datetime.now(), rank, i + 1, len(jobs)))
    x0 = fit_data["thetas"][n, :]
    # Residuals of data
    dy_data = log_gobs - model.predict(log_gbar, x0)
    # Residuals of mock
    xmock, ymock, __ = frame.make_Xy(
        target="gobs", features="gbar", gen_model=gen_model,
        thetas=numpy.copy(gen_model.x0), seed=n)
    dy_mock = ymock - model.predict(xmock.reshape(-1,), x0)

    # Pre-allocate arrays
    spear = numpy.full((len(features), 2), numpy.nan)
    rand_spear = numpy.full_like(spear, numpy.nan)
    kend = numpy.full_like(spear, numpy.nan)
    rand_kend = numpy.full_like(spear, numpy.nan)

    # Calculate the correlation for each feature
    for j in range(len(features)):
        z = Z[:, j]
        spear[j, :] = spearmanr(dy_data, z, nan_policy="omit")
        rand_spear[j, :] = spearmanr(dy_mock, z, nan_policy="omit")
        kend[j, :] = kendalltau(dy_data, z, nan_policy="omit")
        rand_kend[j, :] = kendalltau(dy_mock, z, nan_policy="omit")

    out = {"spear": spear, "rand_spear": rand_spear,
           "kend": kend, "rand_kend": rand_kend}
    joblib.dump(out, ftemp.format(n))


# Wait for all ranks to finish and then combine results
comm.Barrier()
if rank == 0:
    print("Collecting results...", flush=True)
    # Pre-allocate arrays
    spear = numpy.full((nsplits, len(features), 2), numpy.nan)
    rand_spear = numpy.full_like(spear, numpy.nan)
    kend = numpy.full_like(spear, numpy.nan)
    rand_kend = numpy.full_like(spear, numpy.nan)

    for n in range(nsplits):
        finp = joblib.load(ftemp.format(n))
        spear[n, ...] = finp.get("spear", numpy.nan)
        rand_spear[n, ...] = finp.get("rand_spear", numpy.nan)
        kend[n, ...] = finp.get("kend", numpy.nan)
        rand_kend[n, ...] = finp.get("rand_kend", numpy.nan)
        # Remove the temporary file
        remove(ftemp.format(n))

    # Save all results
    out = {"spear": spear, "rand_spear": rand_spear,
           "kend": kend, "rand_kend": rand_kend, "features": features}
    print("Saving to.. `{}`.".format(fperm), flush=True)
    joblib.dump(out, fperm)
