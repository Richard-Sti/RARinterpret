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

from os.path import join
import joblib
import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from scipy.stats import norm

import utils

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


###############################################################################
#                            Partial correlations                             #
###############################################################################


def make_pc(model, kind):
    print(f"Plotting partial correlations for {model}.", flush=True)
    data = joblib.load("../results/pc_{}.p".format(model))

    # Order according to correlation strength
    ordering = numpy.argsort(
        numpy.abs(numpy.mean(data[kind][:, :, 0], axis=0)))[::-1]
    rhos = [data[kind][:, i, 0] for i in ordering]
    rand_rhos = [data["rand_{}".format(kind)][:, i, 0] for i in ordering]
    features = [data["features"][i] for i in ordering]
    xticks = numpy.arange(1, len(features) + 1)
    quantiles = [norm.cdf(x=[-2, -1, 0, 1, 2])] * len(features)

    with plt.style.context(["science"]):
        plt.figure()
        plt.scatter(xticks, [numpy.mean(rho) for rho in rhos], c="red",
                    marker="x", s=15)
        plt.violinplot(rand_rhos, positions=xticks, quantiles=quantiles,
                       showextrema=False)
        plt.axhline(0, ls="--", c="black",  lw=plt.rcParams["axes.linewidth"])
        plt.xticks(xticks,
                   RARinterpret.pretty_label(features, RARinterpret.names),
                   rotation=45)
        plt.ylabel(r"$\tau(g_{\rm obs}, x_k | g_{\rm bar})$")

        plt.tight_layout()

        for ext in ["pdf", "png"]:
            fout = join(utils.plotdir, f"pcs_{model}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


def make_ferr_scaling():
    print("Plotting error scaling.", flush=True)
    d = joblib.load("../results/ferr_scaling.p")
    pars = d["pars"]
    scales = d["scales"]
    rhos = d["rhos"]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    with plt.style.context(["science"]):
        plt.figure()
        for i, par in enumerate(pars):
            mu = numpy.mean(rhos[i, ...], axis=-1)
            std = numpy.std(rhos[i, ...], axis=-1)
            plt.plot(scales, mu, c=cols[i],
                     label=RARinterpret.pretty_label(par, RARinterpret.names))
            plt.fill_between(scales, mu - std, mu + std, color=cols[i],
                             alpha=0.2)

        plt.axhline(0, c="black", ls="--")
        plt.legend(ncol=4, loc="upper center", handlelength=1.5,
                   labelspacing=0.3, columnspacing=1)
        plt.xlabel(r"$f_{\rm err}$")
        plt.ylabel(r"$\tau(g_{\rm obs}, e_{\rm N} | g_{\rm bar})$")
        plt.xlim(0, 1)

        plt.tight_layout()

        for ext in ["pdf", "png"]:
            fout = join(utils.plotdir, f"error_scaling.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


def make_pc_new(fit_model):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(f"Plotting new V2 partial correlations for {fit_model}.", flush=True)
    file = "../results/pc_{}_{}.p"

    data_sparc = joblib.load(file.format(fit_model, "SPARC"))
    ordering = numpy.argsort(numpy.abs(numpy.mean(data_sparc["corr"][..., 0], axis=0)))[::-1]  # noqa
    features = numpy.asanyarray(data_sparc["features"])[ordering]

    xticks = numpy.arange(1, len(features) + 1)
    quantiles = [norm.cdf(x=[-2, -1, 1, 2])] * len(features)

    with plt.style.context(["science"]):
        plt.figure()

        # SPARC
        plt.scatter(xticks,
                    numpy.median(data_sparc["corr"][..., 0], axis=0)[ordering],
                    c="red", marker="x", s=15, label="SPARC", zorder=10)

        # SIFEFE mocks
        rhos = joblib.load(file.format(fit_model, "SIFEFE"))["corr"][:, ordering, 0]  # noqa
        plt.scatter(xticks - 0.05, numpy.median(rhos, axis=0),
                    marker="_", s=15, c=cols[0], label=r"Simple IF + EFE")
        plt.violinplot(rhos, positions=xticks - 0.05, quantiles=quantiles,
                       showextrema=False)

        # SB-Jobs mocks
        rhos = joblib.load(file.format(fit_model, "SB-Jobs"))["corr"][:, ordering, 0]  # noqa
        rhos_med = numpy.median(rhos, axis=0)
        ylower = rhos_med - numpy.percentile(rhos, 1e2 * norm.cdf(-2), axis=0)
        yupper = numpy.percentile(rhos, 1e2 * norm.cdf(2), axis=0) - rhos_med
        yerr = numpy.vstack([ylower, yupper])
        plt.errorbar(xticks + 0.05, rhos_med, yerr=yerr, fmt=" ", marker="o",
                     label=r"$\Sigma_{\rm tot} - J_{\rm obs}$", zorder=-1,
                     ms=2, c=cols[2])

        plt.axhline(0, ls="--", c="black",  lw=plt.rcParams["axes.linewidth"])
        plt.xticks(xticks,
                   RARinterpret.pretty_label(features, RARinterpret.names),
                   rotation=45)
        plt.ylabel(r"$\tau(g_{\rm obs}, x_k | g_{\rm bar})$")
        plt.legend(ncols=3, loc="upper right", fontsize="x-small",
                   columnspacing=0.2, handletextpad=0.1)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"pcs_new_{fit_model}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def make_pc_sb_jobs(fit_model):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(f"Plotting new V2 partial correlations for {fit_model}.", flush=True)
    file = "../results/pc_sbjobs_{}_{}.p"

    data_sparc = joblib.load(file.format(fit_model, "SPARC"))
    ordering = numpy.argsort(numpy.abs(numpy.mean(data_sparc["corr"][..., 0], axis=0)))[::-1]  # noqa
    features = numpy.asanyarray(data_sparc["features"])[ordering]

    xticks = numpy.arange(1, len(features) + 1)
    quantiles = [norm.cdf(x=[-2, -1, 1, 2])] * len(features)

    with plt.style.context(["science"]):
        plt.figure()

        plt.scatter(xticks,
                    numpy.median(data_sparc["corr"][..., 0], axis=0)[ordering],
                    c="red", marker="x", s=15, label="SPARC", zorder=10)
        label = r"$\Sigma_{\rm tot} - J_{\rm obs}$"
        gen_model = "SB-Jobs"
        rhos = joblib.load(file.format(fit_model, gen_model))["corr"][:, ordering, 0]  # noqa
        plt.scatter(xticks, numpy.median(rhos, axis=0), marker="_", s=15,
                    c=cols[0], label=label)
        plt.violinplot(rhos, positions=xticks, quantiles=quantiles,
                       showextrema=False)

        plt.axhline(0, ls="--", c="black",  lw=plt.rcParams["axes.linewidth"])
        plt.xticks(xticks,
                   RARinterpret.pretty_label(features, RARinterpret.names),
                   rotation=45)
        plt.ylabel(r"$\tau(g_{\rm obs}, x_k | g_{\rm bar})$")
        plt.legend(ncols=3, loc="upper right", fontsize="x-small",
                   columnspacing=0.2, handletextpad=0.1)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"pcs_sbjobs_{fit_model}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # make_pc("IF", "kend")
    # make_ferr_scaling()

    make_pc_new("RARIF")
    # make_pc_sb_jobs("SB-Jobs")

    # out = {"loss": loss, "corr": corr, "features": parser_args.features}
    # print("Saving to.. `{}`.".format(fperm), flush=True)
    # joblib.dump(out, fperm)
