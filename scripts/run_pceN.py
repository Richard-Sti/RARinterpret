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
from copy import deepcopy
from os import remove

import numpy
from joblib import dump
from mpi4py import MPI
from scipy.optimize import minimize
from scipy.stats import kendalltau

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
verbose = rank == 0

# Set up the models and data
frame = RARinterpret.RARFrame()
frame_copy = deepcopy(frame)

gen_model = RARinterpret.SimpleIFEFE()
fit_model = RARinterpret.RARIF()
var_loggbar = frame.generate_log_variance("gbar")
var_loggobs = frame.generate_log_variance("gobs")
logeN = frame["log_eN_noclust"]
pars = ["inc", "dist", "Vobs", "L36"]
fperm = "../results/ferr_scaling.p"
ftemp = "../results/temp_ferr_scaling_{}.npy"
nresample = 1000
scales = numpy.linspace(1e-8, 1, 50)


def fit(frame_local):
    """
    Generate mock data, fit the model and return the correlation coefficient.
    """
    mock_x, mock_y, __ = frame_local.make_Xy(
        target="gobs", features="gbar", gen_model=gen_model,
        thetas=numpy.copy(gen_model.x0))
    mask = ~numpy.isnan(mock_y)
    mock_x = mock_x.reshape(-1,)[mask]
    mock_y = mock_y[mask]

    args = (mock_x, mock_y, var_loggbar[mask], var_loggobs[mask],)
    res = minimize(
        fit_model, fit_model.x0 * numpy.random.uniform(0.9, 1.1),
        args=args, method="Nelder-Mead", options={"maxiter": 1000})
    return kendalltau(mock_y - fit_model.predict(mock_x, res.x),
                      logeN[mask])[0]


njobs = len(RARinterpret.split_jobs(nresample, nproc)[rank])
rho = numpy.full((len(pars), scales.size, njobs), numpy.nan)
for i, par in enumerate(pars):
    for j, scale in enumerate(scales):
        if verbose:
            print(f"Calculating parameter {par}, scale {j + 1}/{scales.size}.",
                  flush=True)
        for x in ["inc", "dist", "Vobs", "L36"]:
            frame_copy._data[f"e_{x}"] = frame[f"e_{x}"]
            frame_copy._data[f"e_{x}"] *= (scale if par == x else 1e-8)
        # Now do the resampling with MPI.
        for k in range(njobs):
            rho[i, j, k] = fit(frame_copy)


# Now gather the results and save them.
numpy.save(ftemp.format(rank), rho)
comm.Barrier()
if rank == 0:
    rhos = []
    for i in range(nproc):
        rhos.append(numpy.load(ftemp.format(i)))
        remove(ftemp.format(i))
    rhos = numpy.concatenate(rhos, axis=-1)

    d = {"scales": scales, "rhos": rhos, "pars": pars}
    dump(d, fperm)
    print(f"All finished, output saved to {fperm}.")
