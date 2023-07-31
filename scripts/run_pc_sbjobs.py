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
MPI parallelised partial correlation calculation of the SB-Jobs relation.
"""
from argparse import ArgumentParser
from os import remove
from os.path import join

import joblib
import numpy
from astropy import units
from mpi4py import MPI
from scipy.optimize import minimize
from scipy.stats import kendalltau

# Local packages
try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


def get_data(frame, parser_args, seed):
    var_log_sb = frame.generate_log_variance("SB")
    var_log_jobs = frame.generate_log_variance("Jobs")

    if parser_args.gen_model == "SPARC":
        gen_model = None
        mock_kind = "RAR"  # Doesn't matter
    elif parser_args.gen_model == "SB-Jobs":
        gen_model = RARinterpret.SJCubic()
        mock_kind = "SB-Jobs"
    else:
        raise ValueError(f"Unknown gen_model: `{parser_args.gen_model}`.")

    # Sample the mocks
    x0 = numpy.copy(gen_model.x0) if gen_model is not None else None
    log_sb, log_jobs, __ = frame.make_Xy(
        target=(3, 2), features="SB", gen_model=gen_model, thetas=x0,
        mock_kind=mock_kind, seed=seed, )
    log_sb = log_sb.reshape(-1,)

    if parser_args.gen_model == "SPARC":
        log_jobs -= numpy.log10(1e13 / units.kpc.to(units.km))

    # Now get true SPARC data
    X, __, __ = frame.make_Xy(target=(3, 2), features=parser_args.features)

    return {"log_sb": log_sb, "log_jobs": log_jobs,
            "var_log_sb": var_log_sb, "var_log_jobs": var_log_jobs, "X": X}


def get_fit(data, parser_args):
    if parser_args.fit_model == "SB-Jobs":
        fit_model = RARinterpret.SJCubic()
    else:
        raise ValueError(f"Unknown fit model: `{parser_args.fit_model}`.")

    args = (data["log_sb"], data["log_jobs"], data["var_log_sb"],
            data["var_log_jobs"],)
    x0 = numpy.copy(fit_model.x0) * numpy.random.uniform(0.9, 1.1)
    res = minimize(fit_model, x0, args=args, method="Nelder-Mead")
    ypred = fit_model.predict(data["log_sb"], res.x)

    return {"thetas": res.x,
            "residuals": data["log_jobs"] - ypred,
            "loss": fit_model(res.x, *args) / data["log_sb"].size,
            "success": res.success}


def get_corr(fit_data, data, parser_args):
    nfeatures = len(parser_args.features)
    corr = numpy.full((nfeatures, 2), numpy.nan)
    for i in range(nfeatures):
        corr[i, :] = kendalltau(fit_data["residuals"], data["X"][:, i],
                                nan_policy="omit")
    return corr


def main(tasks, parser_args, ftemp, rank, frame):
    ntasks = len(tasks)
    out_loss = numpy.full(ntasks, numpy.nan)
    out_corr = numpy.full((ntasks, len(parser_args.features), 2), numpy.nan)
    print(f"Rank {rank} processing {ntasks} tasks.", flush=True)

    from tqdm import tqdm

    for i, task in enumerate(tqdm(tasks)):
        data = get_data(frame, parser_args, seed=task)
        fit_data = get_fit(data, parser_args)
        corr = get_corr(fit_data, data, parser_args)

        out_loss[i] = fit_data["loss"]
        out_corr[i, ...] = corr

    out = {"loss": out_loss, "corr": out_corr}
    joblib.dump(out, ftemp.format(rank))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fit_model", type=str, choices=["SB-Jobs"])
    parser.add_argument("--gen_model", type=str, choices=["SPARC", "SB-Jobs"])
    parser.add_argument("--features", type=str, nargs="+")
    parser.add_argument("--nrepeat", type=int)
    parser_args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    dumpdir = "/mnt/extraspace/rstiskalek/rar"
    ftemp = join(
        dumpdir, "temp",
        f"pc_sbjobs_{parser_args.fit_model}_{parser_args.gen_model}" + "_{}.p")
    fperm = f"../results/pc_sbjobs_{parser_args.fit_model}_{parser_args.gen_model}.p"  # noqa

    frame = RARinterpret.RARFrame()
    tasks = RARinterpret.split_jobs(parser_args.nrepeat, nproc)

    main(tasks[rank], parser_args, ftemp, rank, frame)

    comm.Barrier()
    if rank == 0:
        print("Collecting results...", flush=True)
        loss = numpy.full(parser_args.nrepeat, numpy.nan)
        corr = numpy.full((parser_args.nrepeat, len(parser_args.features), 2),
                          numpy.nan)
        start = 0
        for n in range(nproc):
            end = start + len(tasks[n])
            finp = joblib.load(ftemp.format(n))
            loss[start:end] = finp.get("loss", numpy.nan)
            corr[start:end, ...] = finp.get("corr", numpy.nan)
            remove(ftemp.format(n))
            start = end

        out = {"loss": loss, "corr": corr, "features": parser_args.features}
        print("Saving to.. `{}`.".format(fperm), flush=True)
        joblib.dump(out, fperm)

    comm.Barrier()
    quit()
