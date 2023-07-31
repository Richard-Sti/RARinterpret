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

import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from scipy.stats import kendalltau
from tqdm import trange, tqdm

import utils

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


def plot_luminosity_vobs():
    print("Plotting L36 - Vobs..", flush=True)
    frame = RARinterpret.RARFrame()
    gen_model = RARinterpret.RARIF()

    nresample = 50
    Xs = numpy.full((nresample, len(frame)), numpy.nan)
    ys = numpy.full((nresample, len(frame)), numpy.nan)
    corrs = numpy.full((nresample), numpy.nan)

    for i in trange(nresample):
        X, y, __ = frame.make_Xy((1, 0), features="L36", gen_model=gen_model,
                                 thetas=gen_model.x0, )
        X = X.reshape(-1, )
        corrs[i] = kendalltau(X, y)[0]
        Xs[i] = 10**X
        ys[i] = 10**y

    Xs = numpy.concatenate(Xs)
    ys = numpy.concatenate(ys)

    with plt.style.context(utils.pltstyle):
        plt.figure()
        plt.scatter(frame["L36"], frame["Vobs"], s=2, c="red", label="SPARC")
        plt.scatter(Xs, ys, s=2, alpha=0.5, label="Mock")

        plt.xlabel(r"$L_{3.6}$")
        plt.ylabel(r"$V_{\rm obs}$")
        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fout = join(utils.plotdir, f"luminosity_to_vobs.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()

    with plt.style.context(utils.pltstyle):
        plt.figure()
        plt.hist(corrs, bins="auto", label=r"Mock realisations")
        plt.axvline(kendalltau(frame["L36"], frame["Vobs"])[0], c="red",
                    label=r"SPARC data")
        plt.xlabel(r"$\tau(L_{3.6}, V_{\rm obs})$")
        plt.ylabel("Counts")
        plt.legend()

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fout = join(utils.plotdir, f"luminosity_to_vobs_corr.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


def plot_luminosity_vobs_realisations():
    print("Plotting L36 - Vobs resamples..", flush=True)
    frame = RARinterpret.RARFrame()
    gen_model = RARinterpret.RARIF()

    nresample = 10
    for i in range(nresample):

        X, y, __ = frame.make_Xy((1, 0), features="L36", gen_model=gen_model,
                                 thetas=gen_model.x0)
        X = 10**X.reshape(-1, )
        y = 10**y

        with plt.style.context(utils.pltstyle):
            plt.figure()
            plt.scatter(frame["L36"], frame["Vobs"], s=2, c="red",
                        label="SPARC")
            plt.scatter(X, y, s=2, label="Mock")

            plt.xlabel(r"$L_{3.6}$")
            plt.ylabel(r"$V_{\rm obs}$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()

            plt.tight_layout()
            for ext in ["png"]:
                fout = join(utils.plotdir,
                            f"luminosity_to_vobs_{str(i).zfill(2)}.{ext}")
                print(f"Saving to `{fout}`.", flush=True)
                plt.savefig(fout, dpi=utils.dpi)
            plt.close()


def plot_sb_vs_jobs():
    print("Plotting SB - Jobs resamples..", flush=True)
    frame = RARinterpret.RARFrame()

    func = RARinterpret.SJCubic()
    xrange = frame["SB"][frame["SB"] > 0]
    xrange = numpy.linspace(numpy.log10(numpy.nanmin(xrange)),
                            numpy.log10(numpy.nanmax(xrange)), 1000)
    ypred = func.predict(xrange, func.x0)

    with plt.style.context(utils.pltstyle):
        plt.figure()
        plt.scatter(frame["SB"], frame["Jobs"], s=2, label="SPARC")
        plt.plot(10**xrange, 10**ypred, c="red", label="Cubic fit")

        plt.xlabel(r"$\Sigma_{\rm tot}$")
        plt.ylabel(r"$J_{\rm obs}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.plotdir, f"sb_vs_jobs.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


# def mock_sb_jobs():
#     frame = RARinterpret.RARFrame()
#     func = RARinterpret.SJCubic()
#     xrange = frame["SB"][frame["SB"] > 0]
#     xrange = numpy.linspace(numpy.log10(numpy.nanmin(xrange)),
#                             numpy.log10(numpy.nanmax(xrange)), 1000)
#     ypred = func.predict(xrange, func.x0)
#
#     target = (3, 2)
#     gen_model = RARinterpret.SJCubic()
#     thetas = gen_model.x0
#     mock_kind = "SB-Jobs"
#
#     X, y, features = frame.make_Xy(target, gen_model=gen_model,
#                                    thetas=thetas, mock_kind=mock_kind)
#
#     k = features.index("SB")
#     with plt.style.context(utils.pltstyle):
#         plt.figure()
#         plt.scatter(frame["SB"], frame["Jobs"], s=2, label="SPARC")
#         plt.scatter(10**X[:, k], 10**y, s=2, label="Mock")
#         plt.xlabel(r"$\Sigma_{\rm tot}$")
#         plt.ylabel(r"$J_{\rm obs}$")
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.xlim(0.025)
#         plt.legend()
#
#         plt.tight_layout()
#         for ext in ["png", "pdf"]:
#             fout = join(utils.plotdir, f"mock_sb_vs_jobs.{ext}")
#             print(f"saving to `{fout}`.", flush=True)
#             plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
#         plt.close()

def get_predgobs(frame, kind):
    if kind == "disk":
        gen_model = RARinterpret.ToyExponentialModel()
        pred_sb = gen_model.surface_brightness(frame["r"], frame["Rdisk0"],
                                               frame["SBdisk0"])
        pred_gbar = gen_model(frame["r"], frame["Rdisk0"], frame["SBdisk0"])
    elif kind == "sph":
        gen_model = RARinterpret.ToyExponentialModel()
        pred_sb = gen_model.surface_brightness(frame["r"], frame["Rdisk0"],
                                               frame["SBdisk0"])
        pred_gbar = gen_model.gbar_spherical(frame["r"], frame["Rdisk0"],
                                             frame["SBdisk0"])
    elif kind == "kuzmin":
        gen_model = RARinterpret.ToyKuzminModel()
        mtot = frame["L36"] * 1e9
        a = frame["Rdisk0"] / 10
        pred_sb = gen_model.surface_brightness(frame["r"], mtot, a)
        pred_gbar = gen_model(frame["r"], mtot, a)
    else:
        raise ValueError(f"Unrecognized kind `{kind}`.")
    rarmodel = RARinterpret.RARIF()
    pred_gobs = 10**rarmodel.predict(numpy.log10(pred_gbar), rarmodel.x0)
    return pred_sb, pred_gbar, pred_gobs


def plot_gencomb_sb():
    frame = RARinterpret.RARFrame()

    pred_sb_disk, pred_gbar_disk, pred_gobs_disk = get_predgobs(frame, "disk")  # noqa
    pred_sb_sph, pred_gbar_sph, pred_gobs_sph = get_predgobs(frame, "sph")  # noqa
    pred_sb_kuzm, pred_gbar_kuzm, pred_gobs_kuzm = get_predgobs(frame, "kuzmin")  # noqa

    thetas = numpy.linspace(0, numpy.pi, 100)
    corr = numpy.full((thetas.size, 3), numpy.nan)
    for i, theta in enumerate(tqdm(thetas)):
        alpha = numpy.cos(theta)
        beta = numpy.sin(theta)

        D = pred_gobs_disk**(alpha / 2) / frame["r"]**(beta - alpha / 2)
        corr[i, 0] = kendalltau(pred_sb_disk, D)[0]

        D = pred_gobs_sph**(alpha / 2) / frame["r"]**(beta - alpha / 2)
        corr[i, 1] = kendalltau(pred_sb_sph, D)[0]

        D = pred_gobs_kuzm**(alpha / 2) / frame["r"]**(beta - alpha / 2)
        corr[i, 2] = kendalltau(pred_sb_kuzm, D)[0]

    with plt.style.context(utils.pltstyle):
        plt.figure()
        plt.scatter(frame["gbar"], pred_gbar_disk, s=0.1, label="Disk")
        plt.scatter(frame["gbar"], pred_gbar_sph, s=0.1, label="``Spherical''")
        plt.scatter(frame["gbar"], pred_gbar_kuzm, s=0.1, label="Kuzmin")
        t = numpy.linspace(1e-2, 1e2, 100)
        plt.plot(t, t, c="k", linewidth=0.8, zorder=0, label="1:1")

        plt.legend()
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel(r"True $g_{\rm bar}$")
        plt.ylabel(r"Predicted $g_{\rm bar}$")
        plt.legend(ncols=2, fontsize="small", columnspacing=0.2,
                   handletextpad=0.2)

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.plotdir, f"pred_gbar.{ext}")
            print(f"saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()

        with plt.style.context(utils.pltstyle):
            plt.figure()
            plt.scatter(frame["SB"], pred_sb_disk, s=0.1, label="Disk")
            plt.scatter(frame["SB"], pred_sb_sph, s=0.1, label="Spherical")
            plt.scatter(frame["SB"], pred_sb_kuzm, s=0.1, label="Kuzmin")
            plt.plot(t, t, c="k", linewidth=0.8, zorder=0, label="1:1")

            plt.legend()
            plt.xscale("log")
            plt.yscale("log")

            plt.xlabel(r"True $\Sigma_{\rm tot}$")
            plt.ylabel(r"Predicted $\Sigma_{\rm disk}$")
            plt.legend(ncols=2, fontsize="small")

            plt.tight_layout()
            for ext in ["png"]:
                fout = join(utils.plotdir, f"pred_sb.{ext}")
                print(f"saving to `{fout}`.", flush=True)
                plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
            plt.close()

    with plt.style.context(utils.pltstyle):
        plt.figure()
        plt.plot(thetas, corr[:, 0], label="Disk")
        plt.plot(thetas, corr[:, 1], label="Spherical")
        plt.plot(thetas, corr[:, 2], label="Kuzmin")
        plt.axvline(numpy.arctan(1 / 2), c="cyan", label=r"$g_{\rm obs}$",
                    linewidth=0.8, zorder=0)
        plt.axvline(numpy.arctan(2 / 3), c="black", label=r"$J_{\rm obs}$",
                    linewidth=0.8, zorder=0)

        plt.xlabel(r"$\theta~[\mathrm{rad}]$")
        plt.ylabel(r"$\tau (\mathcal{D}, \Sigma)$")
        plt.legend(ncols=2, fontsize="small")

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.plotdir, f"gencomb_sb.{ext}")
            print(f"saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def test_plot():
    frame = RARinterpret.RARFrame()

    gen_model = RARinterpret.ToyExponentialModel()
    # rarmodel = RARinterpret.RARIF()

    pred_gbar = gen_model.gbar_disk(frame["r"], frame["Rdisk0"],
                                    frame["SBdisk0"])

    pred_gbar_old = gen_model(frame["r"], frame["Rdisk0"], frame["SBdisk0"])

    with plt.style.context(utils.pltstyle):

        plt.figure()

        plt.scatter(frame["gbar"], pred_gbar, s=0.1, label="Disk")
        plt.scatter(frame["gbar"], pred_gbar_old, s=0.1, label="Spherical")

        t = numpy.linspace(1e-2, 1e2, 1000)
        plt.plot(t, t, c="red", zorder=0, label="1:1")

        # plt.hist(y, bins="auto")
        # plt.legend()

        plt.xlabel(r"True $g_{\rm bar}$")
        plt.ylabel(r"Predicted $g_{\rm bar}$")

        plt.legend()

        plt.xscale("log")
        plt.yscale("log")

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.plotdir, f"test.{ext}")
            print(f"saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def plot_lines():
    frame = RARinterpret.RARFrame()
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # gen_model = RARinterpret.ToyExponentialModel()
    # rarmodel = RARinterpret.RARIF()
    # gen_model = RARinterpret.ToyExponentialModel()
    # pred_gbar = gen_model.gbar_spherical(frame["r"], frame["Rdisk0"],
    #                                      frame["SBdisk0"])
    # pred_sb = gen_model.surface_brightness(frame["r"], frame["Rdisk0"],
    #                                        frame["SBdisk0"])
    # pred_gobs = 10**rarmodel.predict(numpy.log10(pred_gbar), rarmodel.x0)

    # logr = numpy.log10(frame["r"])

    # for n in numpy.unique(frame["index"]):
    #     mask = numpy.sum(frame["index"] == n)
    #     if mask > 50:
    #         print(n, mask.sum())

    m = frame["index"] == 93
    with plt.style.context(utils.pltstyle):

        plt.figure()

        plt.plot(frame["r"][m], frame["SB"][m], label=r"$\Sigma$", zorder=1)
        for n in range(1, 5):
            alpha = n
            beta = n - 1

            D = frame["gobs"]**(alpha / 2) / frame["r"]**(beta - alpha / 2)
            plt.plot(frame["r"][m], D[m] * 100,
                     label=r"$\mathcal{{D}}({}, {})$".format(alpha, beta),
                     zorder=0, c=cols[n])
            # ps = numpy.polyfit(logr, numpy.log10(D), 2)
            # m = numpy.argsort(frame["r"])
            # plt.plot(frame["r"][m], 10**(numpy.polyval(ps, logr[m])),
            #          c=cols[n], zorder=2)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$r~[\mathrm{kpc}]$")

        plt.legend(ncols=2, markerscale=5, fontsize="small")

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.plotdir, f"test.{ext}")
            print(f"saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # plot_luminosity_vobs()
    # plot_luminosity_vobs_realisations()
    # plot_sb_vs_jobs()
    #  mock_sb_jobs()
    # plot_sb_vs_vobs()
    # test_plot_corr()
    # plot_lines()
    plot_gencomb_sb()
