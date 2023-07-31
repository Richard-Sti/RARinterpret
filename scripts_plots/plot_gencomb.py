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

import utils

try:
    import RARinterpret
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import RARinterpret


###############################################################################
#                              D(alpha, beta) plot                            #
###############################################################################

def read(reg, feature, isdata):
    if isdata:
        d = joblib.load(f"../results/gencomb/{reg}_{feature}.p")
    else:
        d = joblib.load(f"../results/gencomb/mock_{reg}_{feature}.p")

    if reg == "ET":
        d["scores"] /= 2693  # TODO later remove
    mu = numpy.mean(d["scores"], axis=-1)
    std = numpy.std(d["scores"], axis=-1)
    return d["thetas"], mu, std


def make_gencomb(reg, features):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(figsize=(3.5, 2.625 * 1.15))

        for i, feature in enumerate(features):
            xrange, mu, std = read(reg, feature, isdata=True)
            if feature == "MHI":
                mu *= 2693  # TODO: later remove
                std *= 2693
            label = RARinterpret.pretty_label(feature, RARinterpret.names)
            ax.plot(xrange, mu, c=cols[i % len(cols)], label=label)
            ax.fill_between(xrange, mu - std, mu + std,
                            color=cols[i % len(cols)], alpha=0.1)

            try:
                __, mock_mu, __ = read(reg, feature, isdata=False)
                mu = numpy.mean(mock_mu, axis=-1)
                # std = numpy.std(mock_mu, axis=-1)
                ax.plot(xrange, mu, c=cols[i % len(cols)], ls="dashed")
                # ax.fill_between(xrange, mu - std, mu + std,
                #                 color=cols[i % len(cols)], alpha=0.4, hatch="////")  # noqa
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

        try:
            mu, mean, std = read(reg, "gbar,SB,MHI", True)  # noqa
            # TODO: later remove
            mean *= 2693
            std *= 2693
            col = cols[(len(features) + 1) % len(cols)]
            ax.plot(xrange, mean, c=col,
                    label=r"$g_{\rm bar},\,\Sigma_{\rm tot},\,M_{\rm HI}$")
            ax.fill_between(xrange, mean - std, mean + std, color=col,
                            alpha=0.1)
        except FileNotFoundError:
            print(f"No `gbar,SB,MHI` data found for {reg}.")

        try:
            mu, mean, std = read(reg, "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust", True)  # noqa
            # TODO: later remove
            mean *= 2693
            std *= 2693
            ax.plot(xrange, mean, c="black", label=r"$\mathrm{All}$")
            ax.fill_between(xrange, mean - std, mean + std, color="black",
                            alpha=0.1)
        except FileNotFoundError:
            print(f"No `all` data found for {reg}.")

        try:
            __, mock_mu, __ = read(reg, "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust", isdata=False)  # noqa
            mock_mu *= 2693
            # TODO: later remove
            mu = numpy.mean(mock_mu, axis=-1)
            # std = numpy.std(mock_mu, axis=-1)
            ax.plot(xrange, mu, c="black", ls="dashed")
            # ax.fill_between(xrange, mu - std, mu + std, color="black",
            #                 alpha=0.3, hatch="x")
        except FileNotFoundError:
            print(f"No `all` mock data found for {reg}.")

        for k in range(3):
            plt.axvline(numpy.arctan(k / (k + 1)), c="black", ls="dotted",
                        zorder=0)

        secax = ax.secondary_xaxis('top')
        # Force offset gobs and Jobs labels
        secax.set_xticks([numpy.arctan(0), numpy.arctan(1 / 2) - 0.05,
                          numpy.arctan(2 / 3) + 0.05])
        secax.set_xticklabels(
            [r"$V_{\rm obs}$", r"$g_{\rm obs}$", r"$J_{\rm obs}$"],
            rotation=70)

        ax.set_xlim(0, numpy.pi)
        ax.set_xlabel(r"$\theta~[\mathrm{rad}]$")
        if reg == "ET":
            ax.set_ylabel(r"Loss $\mathcal{L}_0$ per observation")
        else:
            ax.set_ylabel(r"Loss $\mathcal{L}$ per observation")

        if reg == "NN":
            ax.set_ylim(0.3)

        ax.legend(loc="upper left", ncol=3, columnspacing=0.25, handlelength=1,
                  bbox_to_anchor=(0.2, 1.25))

        ax.set_yscale("log")
        fig.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"gencomb_{reg}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi)
        plt.close()


###############################################################################
###############################################################################
###############################################################################

def make_imp():
    features = ["gbar", "SB", "L36", "type", "MHI"]
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    f2col = {f: cols[i % len(cols)] for i, f in enumerate(features)}

    data = joblib.load("../results/gencomb/ET_gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust.p")  # noqa
    mu_sparc = numpy.mean(data["imps"][..., 1], axis=1)
    std_sparc = numpy.std(data["imps"][..., 1], axis=1)
    norm = numpy.sum(mu_sparc, axis=1)
    mu_sparc /= norm.reshape(-1, 1)
    std_sparc /= norm.reshape(-1, 1)


    mock_rar  = joblib.load("../results/gencomb/mock_ET_gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust.p")  # noqa
    imps_mock = numpy.mean(mock_rar["imps"], axis=1)
    mu_rar = numpy.mean(imps_mock[..., 1], axis=1)
    norm = numpy.sum(mu_rar, axis=1)
    mu_rar /= norm.reshape(-1, 1)

    mock_sbjobs = joblib.load("../results/gencomb/mock_ETSB-Jobs_gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust.p")  # noqa
    imps_mock = numpy.mean(mock_sbjobs["imps"], axis=1)
    mu_sbjobs = numpy.mean(imps_mock[..., 1], axis=1)
    norm = numpy.sum(mu_sbjobs, axis=1)
    mu_sbjobs /= norm.reshape(-1, 1)

    xrange = data["thetas"]
    with plt.style.context(["science"]):
        fig, ax = plt.subplots(
            nrows=2, ncols=1, figsize=(3.5, 2.625 * 1.5), sharex=True,
            gridspec_kw={"height_ratios": [1, 0.5]})
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[1].axhline(0, c=cols[0], ls="solid", zorder=0, label="SPARC",
                      linewidth=0.6)
        for i, feature in enumerate(data["features"]):
            if data["features"][i] not in features:
                continue
            # Data
            label = RARinterpret.pretty_label(feature, RARinterpret.names)
            ax[0].plot(xrange, mu_sparc[:, i], c=f2col[feature], label=label)
            ax[0].fill_between(xrange, mu_sparc[:, i] - std_sparc[:, i],
                               mu_sparc[:, i] + std_sparc[:, i],
                               color=f2col[feature], alpha=0.15)
            # Mock RAR
            ax[0].plot(xrange, mu_rar[:, i], c=f2col[feature], ls="dashed")
            label = "Mock RAR" if i == 0 else None
            ax[1].plot(xrange, mu_rar[:, i] - mu_sparc[:, i], c=f2col[feature],
                       ls="dashed", label=label)
            # Mock SB-Jobs
            ax[0].plot(xrange, mu_sbjobs[:, i], c=f2col[feature], ls="dotted")
            label = r"Mock $\Sigma_{\rm tot} - J_{\rm obs}$" if i == 0 else None  # noqa
            ax[1].plot(xrange, mu_sbjobs[:, i] - mu_sparc[:, i],
                       c=f2col[feature], ls="dotted", label=label)

        ax[1].axhline(0, c="lightsteelblue", ls="solid", zorder=0,
                      linewidth=0.6)
        for i in range(2):
            for k in range(3):
                ax[i].axvline(numpy.arctan(k / (k + 1)), c="lightsteelblue",
                              ls="solid", zorder=0, linewidth=0.4)
            ax[i].set_xlim(0, numpy.pi)

        secax = ax[0].secondary_xaxis('top')
        secax.set_xticks([numpy.arctan(0), numpy.arctan(1 / 2) - 0.05,
                          numpy.arctan(2 / 3) + 0.05])
        secax.set_xticklabels(
            [r"$V_{\rm obs}$", r"$g_{\rm obs}$", r"$J_{\rm obs}$"],
            rotation=70)

        ax[0].set_ylabel(r"Rel. permutation importance")
        ax[1].set_ylabel(r"Mock - SPARC")
        ax[0].legend(loc="upper right", ncol=3, columnspacing=0.1,
                     handlelength=1.5, fontsize="small", handleheight=1.0)
        ax[1].legend(loc="upper right", ncol=2, columnspacing=0.1,
                     handlelength=1.5, fontsize="x-small", handleheight=1.0)
        ax[0].set_ylim(1e-5, 1)
        ax[1].set_xlabel(r"$\theta~[\mathrm{rad}]$")

        fig.tight_layout(h_pad=0.0)
        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"perm_imp.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            fig.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
###############################################################################
###############################################################################


def make_label(label):
    if ',' not in label:
        return RARinterpret.pretty_label(label, RARinterpret.names)
    label = label.split(",")
    label = RARinterpret.pretty_label(label, RARinterpret.names)
    return ", ".join(label)


def make_gencomb_shared(reg, left_features, right_features):
    print("Making gencomb shared plot.", flush=True)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    f2col = {f: cols[i % len(cols)]
             for i, f in enumerate(left_features + right_features)}

    new_features = ["MHI", "gbar,SB,MHI", "gbar,SB"]

    with plt.style.context(["science"]):
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(2 * 3.5, 2.625 * 1.75), sharey="row",
            sharex=True, gridspec_kw={"height_ratios": [1, 2/3]})
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[1, 0].axhline(1, c=cols[0], ls="solid", zorder=-1, linewidth=0.6,
                         label="SPARC")

        # Left panel. Includes mocks.
        for i, feature in enumerate(left_features):
            xrange, mu_sparc, std_sparc = read(reg, feature, isdata=True)
            if feature in new_features:
                mu_sparc *= 2693  # TODO: later remove
                std_sparc *= 2693
            ax[0, 0].plot(xrange, mu_sparc, c=f2col[feature],
                          label=make_label(feature))
            ax[0, 0].fill_between(xrange, mu_sparc - std_sparc,
                                  mu_sparc + std_sparc, color=f2col[feature],
                                  alpha=0.15)

            # Mock RAR
            try:
                __, mock_mu, __ = read(reg, feature, isdata=False)
                mu = numpy.mean(mock_mu, axis=-1)
                if feature in new_features:
                    mu *= 2693  # TODO: later remove
                label = "Mock RAR" if i == 0 else None
                ax[0, 0].plot(xrange, mu, c=f2col[feature], ls="dashed")
                ax[1, 0].plot(xrange, mu / mu_sparc, c=f2col[feature],
                              ls="dashed", label=label)
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

            # Mock SB-Jobs
            try:
                __, mock_mu, __ = read(reg + "SB-Jobs", feature, isdata=False)
                mu = numpy.mean(mock_mu, axis=-1)
                if feature in new_features:
                    mu *= 2693  # TODO: later remove
                label = r"Mock $\Sigma_{\rm tot} - J_{\rm obs}$" if i == 0 else None  # noqa
                ax[0, 0].plot(xrange, mu, c=f2col[feature], ls="dotted")
                ax[1, 0].plot(xrange, mu / mu_sparc, c=f2col[feature],
                              ls="dotted", label=label)
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

        # Right panel. No mocks and uncertainty
        for i, feature in enumerate(right_features):
            xrange, mu_sparc, std_sparc = read(reg, feature, isdata=True)
            if feature in new_features:
                mu_sparc *= 2693  # TODO: later remove
                std_sparc *= 2693
            ax[0, 1].plot(xrange, mu_sparc, c=f2col[feature],
                          label=make_label(feature))

            # Mock RAR
            try:
                __, mock_mu, __ = read(reg, feature, isdata=False)
                mu = numpy.mean(mock_mu, axis=-1)
                if feature in new_features and feature != "MHI":
                    mu *= 2693  # TODO: later remove
                ax[0, 1].plot(xrange, mu, c=f2col[feature], ls="dashed")
                ax[1, 1].plot(xrange, mu / mu_sparc, c=f2col[feature],
                              ls="dashed")
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

            # Mock SB-Jobs
            try:
                __, mock_mu, __ = read(reg + "SB-Jobs", feature, isdata=False)
                mu = numpy.mean(mock_mu, axis=-1)
                ax[0, 1].plot(xrange, mu, c=f2col[feature], ls="dotted")
                ax[1, 1].plot(xrange, mu / mu_sparc, c=f2col[feature],
                              ls="dotted")
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

        # All features
        try:
            mu, mean_sparc, std_sparc = read(reg, "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust", True)  # noqa
            # TODO: later remove
            mean_sparc *= 2693
            std_sparc *= 2693
            ax[0, 1].plot(xrange, mean_sparc, c="black",
                          label=r"$\mathrm{All}$")
        except FileNotFoundError:
            print(f"No `all` data found for {reg}.")

        # Mock RAR
        try:
            __, mock_mu, __ = read(reg, "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust", isdata=False)  # noqa
            mock_mu *= 2693
            # TODO: later remove
            mu = numpy.mean(mock_mu, axis=-1)
            ax[0, 1].plot(xrange, mu, c="black", ls="dashed")
            ax[1, 1].plot(xrange, mu / mean_sparc, c="black", ls="dashed")
        except FileNotFoundError:
            print(f"No `all` mock RAR data found for {reg}.")

        # Mock SB-Jobs
        try:
            __, mock_mu, __ = read(reg + "SB-Jobs", "gbar,SB,dist,inc,L36,MHI,type,Reff,log_eN_noclust", isdata=False)  # noqa
            # mock_mu *= 2693
            # TODO: later remove
            mu = numpy.mean(mock_mu, axis=-1)
            ax[0, 1].plot(xrange, mu, c="black", ls="dotted")
            ax[1, 1].plot(xrange, mu / mean_sparc, c="black", ls="dotted")
        except FileNotFoundError:
            print(f"No `all` mock SB-Jobs data found for {reg}.")

        for i in range(2):
            secax = ax[0, i].secondary_xaxis('top')
            secax.set_xticks([numpy.arctan(0), numpy.arctan(1 / 2) - 0.05,
                              numpy.arctan(2 / 3) + 0.05])
            secax.set_xticklabels(
                [r"$V_{\rm obs}$", r"$g_{\rm obs}$", r"$J_{\rm obs}$"],
                rotation=70)

            ax[1, i].set_xlim(0, numpy.pi)
            ax[1, i].set_xlabel(r"$\theta~[\mathrm{rad}]$")
            ax[0, i].set_yscale("log")

            for k in range(3):
                ax[0, i].axvline(numpy.arctan(k / (k + 1)), c="lightsteelblue",
                                 ls="solid", zorder=0, linewidth=0.6)
                ax[1, i].axvline(numpy.arctan(k / (k + 1)), c="lightsteelblue",
                                 ls="solid", zorder=0, linewidth=0.6)
            ax[1, i].axhline(1, c="lightsteelblue", ls="solid", zorder=0,
                             linewidth=0.6)

        ax[0, 0].legend(loc="lower right", ncol=1, handletextpad=0.2,
                        fontsize="small", columnspacing=0.1, handlelength=1.5,
                        borderaxespad=0.25, handleheight=1.0)
        ax[0, 1].legend(loc="lower right", ncol=2, handletextpad=0.2,
                        fontsize="small", columnspacing=0.1, handlelength=1.5,
                        borderaxespad=0.25, handleheight=1.0)
        ax[1, 0].legend(loc="upper right", ncol=2, handletextpad=0.2,
                        fontsize="small", handlelength=1.5, columnspacing=0.1,
                        borderaxespad=0.25, handleheight=1.0)

        ax[0, 0].set_ylabel(r"Loss $\mathcal{L}_0$ per observation")
        ax[1, 0].set_ylabel(r"$\mathcal{L}_{\rm SPARC} / \mathcal{L}_{\rm mock}$")  # noqa
        ax[1, 0].set_ylim(0.1, 3.2)

        fig.tight_layout(w_pad=0.0, h_pad=0.0)
        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"gencomb_shared{reg}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            fig.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
###############################################################################
###############################################################################


def make_gencomb_corr(reg, features):
    print("Plotting the gencomb correlation plot.")
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    with plt.style.context(["science"]):
        fig, axs = plt.subplots(nrows=2, sharex=True,
                                gridspec_kw={"height_ratios": [1, 1/2]},
                                figsize=[3.5, 2.625 * 1.5])
        fig.subplots_adjust(hspace=0.0, wspace=0.0)
        axs[1].axhline(0, c=cols[0], zorder=-1, label="SPARC",
                       linewidth=0.6)

        for i, feature in enumerate(features):
            labels = RARinterpret.pretty_label(feature, RARinterpret.names)
            xrange, mu_sparc, std_sparc = read(reg, feature, isdata=True)
            axs[0].plot(xrange, mu_sparc, c=cols[i % len(cols)], label=labels)
            axs[0].fill_between(xrange, mu_sparc - std_sparc,
                                mu_sparc + std_sparc,
                                color=cols[i % len(cols)], alpha=0.15)

            # MOCK RAR
            try:
                __, mu_mock, __ = read(reg, feature, isdata=False)
                mu_mock = numpy.mean(mu_mock, axis=-1)
                mu = mu_mock - mu_sparc
                axs[0].plot(xrange, mu_mock, c=cols[i % len(cols)],
                            ls="dashed")
                if feature not in ["type", "MHI"]:
                    label = "Mock RAR" if i == 0 else None
                    axs[1].plot(xrange, mu, c=cols[i % len(cols)], ls="dashed",
                                label=label)
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

            # MOCK SB - Jobs
            try:
                __, mu_mock, __ = read(reg + "SB-Jobs", feature,
                                             isdata=False)
                mu_mock = numpy.mean(mu_mock, axis=-1)
                mu = mu_mock - mu_sparc
                axs[0].plot(xrange, mu_mock, c=cols[i % len(cols)],
                            ls="dotted")
                if feature not in ["type", "MHI"]:
                    label = r"Mock $\Sigma_{\rm tot} - J_{\rm obs}$" if i == 0 else None  # noqa
                    axs[1].plot(xrange, mu, c=cols[i % len(cols)], ls="dotted",
                                label=label)
            except FileNotFoundError:
                print(f"No mock `{feature}` data found for {reg}.")

        for i in range(2):
            for k in range(3):
                axs[i].axvline(numpy.arctan(k / (k + 1)), c="lightsteelblue",
                               ls="solid", zorder=0, linewidth=0.4)
            axs[i].set_xlim(0, numpy.pi)

        axs[1].axhline(0, c="lightsteelblue", zorder=0, linewidth=0.6)

        secax = axs[0].secondary_xaxis('top')
        # Force offset gobs and Jobs labels
        secax.set_xticks([numpy.arctan(0), numpy.arctan(1 / 2) - 0.05,
                          numpy.arctan(2 / 3) + 0.05])
        secax.set_xticklabels(
            [r"$V_{\rm obs}$", r"$g_{\rm obs}$", r"$J_{\rm obs}$"],
            rotation=70)
        axs[1].set_xlabel(r"$\theta~[\mathrm{rad}]$")
        axs[0].set_ylabel(r"$\tau$")
        axs[1].set_ylabel(r"$\tau_{\rm mock} - \tau_{\rm SPARC}$")

        axs[0].legend(loc="upper right", ncol=3, columnspacing=0.1,
                      handlelength=1.5, fontsize="small", borderaxespad=0.05,
                      handleheight=1.0)
        axs[1].legend(loc="lower right", ncol=2, columnspacing=0.1,
                      handlelength=1.5, fontsize="small", borderaxespad=0.05,
                      handleheight=1.0)

        fig.tight_layout(h_pad=0)
        for ext in ["png", "pdf"]:
            fout = join(utils.plotdir, f"gencomb_{reg}.{ext}")
            print(f"Saving to `{fout}`.", flush=True)
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    frame = RARinterpret.RARFrame()

    # print(len(frame))
    # features = ["gbar", "SB", "L36", "type", "MHI"]
    # make_gencomb("ET", features)

    # features = ["gbar", "SB", "L36"]
    # make_gencomb("NN", features)

    make_imp()

    features = ["gbar", "SB", "L36", "type", "MHI"]
    make_gencomb_corr("kend", features)

    left_features = ["gbar", "SB", "L36"]
    right_features = ["type", "MHI", "Reff", "gbar,SB", "gbar,SB,MHI"]
    make_gencomb_shared("ET", left_features, right_features)

    # features = ["gbar", "SB", "L36", "type", "MHI", "Reff"]
    # make_gencomb_corr_sbjobs("kend", features)
