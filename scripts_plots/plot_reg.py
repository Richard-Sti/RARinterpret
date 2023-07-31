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
from copy import deepcopy
from os.path import join, isfile

import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy
import scienceplots  # noqa
from scipy.stats import gaussian_kde
from tqdm import trange, tqdm
from numba import jit

import utils

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


###############################################################################
#                              RAR fit                                        #
###############################################################################


def make_rarfit(frame):
    print("Plotting RAR fit.", flush=True)
    frame = RARinterpret.RARFrame()
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colcount = 0

    def summary_splitypred(ypred, frame):
        xrange = frame["gbar"]
        m = numpy.argsort(xrange)
        mu, std = numpy.mean(ypred, axis=0)[m], numpy.std(ypred, axis=0)[m]
        xrange = xrange[m]
        mu = 10**mu
        std *= mu * numpy.log(10)
        return xrange, mu, std

    with plt.style.context(["science"]):
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                gridspec_kw={"height_ratios": [1, 1/2]},
                                figsize=[3.5, 2.625 * 1.5])
        fig.subplots_adjust(hspace=0)
        axs[0].scatter(frame["gbar"], frame["gobs"], s=0.5, zorder=0,
                       c="dimgrey", rasterized=True)

        for name in ["IF", "EFE"]:
            d = joblib.load(f"../results/analytical/{name}_data_0.4_AD.p")
            xrange, mu, std = summary_splitypred(d["ypred"], frame)
            axs[0].plot(xrange, mu, c=cols[colcount], label=utils.names[name])

            if name == "IF":
                muref = deepcopy(mu)
            axs[1].plot(xrange, mu / muref, c=cols[colcount])
            axs[1].fill_between(xrange, (mu - std) / muref, (mu + std) / muref,
                                color=cols[colcount], alpha=1/5)

            colcount += 1

        # ET & XGB
        for name in ["ET", "XGB", "NN"]:
            # gbar only
            d = joblib.load("../results/fit/{}_gobs_gbar_0.4.p".format(name))
            xrange, mu, std = summary_splitypred(d["ypred"], frame)
            axs[0].plot(xrange, mu, c=cols[colcount], label=utils.names[name])
            axs[0].fill_between(xrange, mu - std, mu + std, alpha=0.2,
                                color=cols[colcount])

            axs[1].plot(xrange, mu / muref, c=cols[colcount])
            axs[1].fill_between(xrange, (mu - std) / muref, (mu + std) / muref,
                                alpha=1/5, color=cols[colcount])
            colcount += 1

        axs[1].axhline(1, ls="--", c="black",
                       lw=plt.rcParams["axes.linewidth"])
        for i in range(2):
            axs[i].set_xscale("log")
        axs[0].set_yscale("log")
        axs[1].set_xlabel(r"$g_{\rm bar}~[10^{-10} \mathrm{m}\mathrm{s}^{-2}]$")  # noqa
        axs[0].set_ylabel(r"$g_{\rm obs}~[10^{-10} \mathrm{m}\mathrm{s}^{-2}]$")  # noqa

        axs[1].set_ylabel(r"$f(g_{\rm bar}) / f_{\mathrm{RAR~IF}} $")
        axs[1].set_ylim(0.7, 1.3)
        axs[0].legend()
        axs[0].set_xlim(numpy.min(frame["gbar"]), numpy.max(frame["gbar"]))
        plt.tight_layout(h_pad=0)

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"RARfit.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            fig.savefig(fout, dpi=utils.dpi)
        plt.close()


###############################################################################
#                            RAR goodness-of-fit                              #
###############################################################################


def make_rarloss():
    print("Plotting RAR loss.", flush=True)
    xrange = numpy.linspace(0.1, 1.5, 1000)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colcount = 0

    def kde_approx(xrange, samples):
        samples /= 2693  # TODO Later remove this normalisation.
        y = gaussian_kde(samples)(xrange)
        y /= numpy.sum(y) * (xrange[1] - xrange[0])
        return xrange, y

    with plt.style.context(["science"]):
        plt.figure()
        for name in ["IF", "EFE"]:
            d = joblib.load(f"../results/analytical/{name}_data_0.4_AD.p")
            plt.plot(*kde_approx(xrange, d["loss"]), label=utils.names[name],
                     c=cols[colcount])
            # TODO remove normalisation later.
            plt.axvline(numpy.mean(joblib.load(f"../results/analytical/{name}_data_0.0_AD.p")["loss"] / 2693),  # noqa
                        ls="dotted", color=cols[colcount])
            colcount += 1

        for name in ["ET", "XGB", "NN"]:
            # gbar only
            d = joblib.load(f"../results/fit/{name}_gobs_gbar_0.4.p")
            if name == "ET":
                d["loss"] *= 2693
            plt.plot(*kde_approx(xrange, d["loss"]), label=utils.names[name],
                     c=cols[colcount])

            # All features
            if name in ["NN", "PINN"]:
                continue
            d = joblib.load(f"../results/fit/{name}_gobs_gbar,r,SBdisk,SBbul,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust_0.4.p")  # noqa
            plt.plot(*kde_approx(xrange, d["loss"]), label=None,
                     c=cols[colcount], ls="dashed")

            colcount += 1

        plt.xlabel(r"Loss $\mathcal{L}$ per observation")
        plt.ylabel("Probability density function")
        plt.legend(loc="upper left", ncols=1, fontsize="small",
                   handletextpad=0.2)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"RARloss.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


###############################################################################
#                            gobs - single feature                            #
###############################################################################


def make_gobs_single():
    print("Plotting gobs - single feature.", flush=True)
    features = ["gbar", "r", "SBdisk", "SBbul", "SB", "dist", "inc", "L36",
                "MHI", "type", "Reff", "log_eN_noclust"]

    def get_loss(reg):
        loss = [None] * len(features)

        for i, feature in enumerate(tqdm(features)):
            d = joblib.load(f"../results/fit/{reg}_gobs_{feature}_0.4.p")
            # TODO remove normalisation later.
            loss[i] = d["loss"] / 2693
        return loss

    loss = get_loss("ET")
    ks = numpy.array([numpy.mean(x) for x in loss]).argsort()
    loss = [loss[k] for k in ks]

    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xticks = numpy.arange(len(features))

    with plt.style.context(["science"]):
        plt.figure()
        plt.violinplot(loss, positions=xticks, showextrema=False,
                       showmedians=False)
        for i in range(len(features)):
            plt.scatter(xticks[i], numpy.median(loss[i]), c=cols[0],
                        marker="_")

        labels = RARinterpret.pretty_label(features, RARinterpret.names)
        plt.xticks(xticks, labels, rotation=45)
        plt.ylabel(r"Loss $\mathcal{L}_0$ per observation")
        for xtick in xticks:
            plt.axvline(xtick, c="black", lw=plt.rcParams["axes.linewidth"],
                        ls="--", alpha=0.25)
        plt.yscale("log")
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"single_features.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


###############################################################################
#                       Plot gbar + xk - gobs                                 #
###############################################################################

def make_gbar_xk_gobs():
    print("Plotting gbar + xk - gobs.", flush=True)
    pairs = ["gbar,r", "gbar,SBdisk", "gbar,SBbul", "gbar,SB", "gbar,dist",
             "gbar,inc", "gbar,L36", "gbar,MHI", "gbar,type", "gbar,Reff",
             "gbar,log_eN_noclust"]

    addfeat = [pair.split(",")[1] for pair in pairs]

    def get_loss(reg):
        loss = [None] * len(pairs)

        for i, pair in enumerate(tqdm(pairs)):
            d = joblib.load(f"../results/fit/{reg}_gobs_{pair}_0.4.p")
            # TODO remove normalisation later.
            loss[i] = d["loss"] / 2693
        return loss

    print("Loading ET loss.", flush=True)
    etloss = get_loss("ET")
    print("Loading XGB loss.", flush=True)
    xgbloss = get_loss("XGB")
    print("Loading NN loss.", flush=True)
    nnloss = get_loss("NN")

    ks = numpy.array([numpy.mean(x) for x in xgbloss]).argsort()
    xgbloss = [xgbloss[k] for k in ks]
    etloss = [etloss[k] for k in ks]
    nnloss = [nnloss[k] for k in ks]
    addfeat = [addfeat[k] for k in ks]

    xgbloss.insert(0, joblib.load("../results/fit/{}_gobs_{}_0.4.p".format("XGB", "gbar"))["loss"] / 2693)  # noqa
    etloss.insert(0, joblib.load("../results/fit/{}_gobs_{}_0.4.p".format("ET", "gbar"))["loss"] / 2693)  # noqa
    nnloss.insert(0, joblib.load("../results/fit/{}_gobs_{}_0.4.p".format("NN", "gbar"))["loss"] / 2693)  # noqa
    addfeat.insert(0, r"$g_{\rm bar}$")

    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    xticks = numpy.arange(len(addfeat))

    with plt.style.context(["science"]):
        plt.figure()
        for loss in [etloss, xgbloss, nnloss]:
            plt.violinplot(loss, positions=xticks, showextrema=False,
                           showmedians=False)

        for i in range(len(addfeat)):
            plt.scatter(xticks[i], numpy.median(etloss[i]), c=cols[0],
                        marker="_", label="ET" if i == 0 else None)
            plt.scatter(xticks[i], numpy.median(xgbloss[i]), c=cols[1],
                        marker="_", label="XGB" if i == 0 else None)
            plt.scatter(xticks[i], numpy.median(nnloss[i]), c=cols[2],
                        marker="_", label="NN" if i == 0 else None)

        plt.axhline(numpy.median(nnloss[0]), ls="--", c=cols[2],
                    lw=plt.rcParams["axes.linewidth"])
        labels = RARinterpret.pretty_label(addfeat, RARinterpret.names)
        for i, label in enumerate(labels):
            if i > 0:
                labels[i] = r"$+$" + label

        plt.xticks(xticks, labels, rotation=45)
        plt.ylabel(r"Loss $\mathcal{L}$ per observation")
        plt.ylim(0.5, 1.7)
        plt.legend(ncol=3)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"feature_pair.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


def make_grid():
    print("Plotting loss grid..", flush=True)

    features = ["gbar", "r", "SBdisk", "SBbul", "SB", "dist", "inc", "L36",
                "type", "Reff", "log_eN_noclust"]
    nfeatures = len(features)

    # Order the diagonal!
    loss = [joblib.load(f"../results/fit/ET_gobs_{f}_0.4.p")["loss"].mean()
            for f in tqdm(features)]
    features = [features[i] for i in numpy.argsort(loss)]

    lossgrid = numpy.full((nfeatures, nfeatures), numpy.nan)
    for i in trange(nfeatures):
        for j in range(nfeatures):
            if i == j:
                pair = features[i]
                file = f"../results/fit/ET_gobs_{pair}_0.4.p"
                d = joblib.load(file)
                lossgrid[i, j] = numpy.mean(d["loss"])
            elif j > i:
                continue
            else:
                pair1 = ",".join([features[i], features[j]])
                pair2 = ",".join([features[j], features[i]])
                file1 = f"../results/fit/ET_gobs_{pair1}_0.4.p"
                file2 = f"../results/fit/ET_gobs_{pair2}_0.4.p"

                if isfile(file1) and isfile(file2):
                    print(f"Both files exist `({file1}, {file2})`!")
                elif isfile(file1):
                    d = joblib.load(file1)
                    lossgrid[i, j] = numpy.mean(d["loss"])
                elif isfile(file2):
                    d = joblib.load(file2)
                    lossgrid[i, j] = numpy.mean(d["loss"])
                else:
                    print(f"File not found for pair `{pair}`.", flush=True)

    with plt.style.context(utils.pltstyle):
        fig, ax = plt.subplots()
        norm = colors.LogNorm(vmin=numpy.nanmin(lossgrid),
                              vmax=numpy.nanmax(lossgrid))
        pcm = ax.imshow(lossgrid, norm=norm, cmap="viridis_r")
        fig.colorbar(pcm, ax=ax, label=r"Loss $\mathcal{L}_0$ per observation")

        labels = RARinterpret.pretty_label(features, RARinterpret.names)
        ax.set_xticks(numpy.arange(nfeatures))
        ax.set_xticklabels(labels, rotation=60)
        ax.set_yticks(numpy.arange(nfeatures))
        ax.set_yticklabels(labels)

        ax.tick_params(axis=u'both', which=u'both', length=0)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"feature_grid.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            fig.savefig(fout, dpi=utils.dpi)
        plt.close()


def outlier_loss(frame, reg="ET"):
    """
    Plot the loss when we remove percentiles of data from the ends.
    """
    print("Plotting RAR outlier loss.", flush=True)
    frame = RARinterpret.RARFrame()

    gradmodel = RARinterpret.RARIF()
    sample_weight = gradmodel.make_weights(
        gradmodel.x0, numpy.log10(frame["gbar"]),
        frame.generate_log_variance("gbar"),
        frame.generate_log_variance("gobs"))
    sample_weight = numpy.asarray(sample_weight)

    data = joblib.load("../results/fit/{}_gobs_gbar_0.4.p".format(reg))
    log_gobs = numpy.log10(frame["gobs"])
    log_gbar = numpy.log10(frame["gbar"])
    ypred_all = data["ypred"]
    test_masks = data["test_masks"]

    @jit(nopython=True)
    def cut_loss(p):
        ntest = test_masks.shape[0]
        loss = numpy.full(ntest, numpy.nan)
        xmin, xmax = numpy.percentile(log_gbar, [p, 100 - p])

        for n in range(ntest):
            test_mask = test_masks[n, :]
            x = log_gbar[test_mask]
            selmask = (x >= xmin) & (x < xmax)

            ytrue = log_gobs[test_mask][selmask]
            ypred = ypred_all[n, test_mask][selmask]
            weight = sample_weight[test_mask][selmask]

            loss[n] = 0.5 * numpy.mean(weight * (ytrue - ypred)**2)
        return numpy.mean(loss), numpy.std(loss)

    xrange = numpy.arange(0, 35)
    y = numpy.full((xrange.size, 2), numpy.nan)
    for i in trange(xrange.size):
        y[i, :] = cut_loss(xrange[i])

    with plt.style.context(["science"]):
        plt.figure()
        plt.plot(xrange, y[:, 0])
        plt.fill_between(xrange, y[:, 0] - y[:, 1], y[:, 0] + y[:, 1],
                         alpha=0.5)
        plt.xlabel("Percentile cut")
        plt.ylabel(r"Loss $\mathcal{L}$ per observation")

        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"RARloss_percentile.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


if __name__ == "__main__":
    frame = RARinterpret.RARFrame()

    make_rarfit(frame)
    # make_rarloss()
    # make_gobs_single()
    # make_gbar_xk_gobs()

    # make_grid()

    # outlier_loss(frame)
