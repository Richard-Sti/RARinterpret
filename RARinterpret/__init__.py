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

from .read import (RARFrame, replace_values, add_columns, rm_columns,  # noqa
                   extract_col_from_X)  # noqa
from .analytical import (RARIF, SimpleIFEFE, BestRARESR, MockRAR, LinearRelation,  # noqa
                         QuadraticRelation, RVRQuadratic, LVQuadratic, SJCubic, MockSBJobs,  # noqa
                         ToyExponentialModel, ToyKuzminModel)  # noqa
from .analysis import (make_test_masks, train_test_from_mask, split_jobs,  # noqa
                       basic_pipeline, get_importance, est_fit_score)  # noqa
from .nn import PLModel  # noqa


def pretty_label(label, pretty_dict):
    """
    Convert labels into their LaTeX counterparts from `pretty_dict`. If a label
    is not in `pretty_dict` simply returns itself.

    Parameters
    ----------
    label : str or a list of str
        Labels to be converted.
    pretty_dict : dict
        Dictionary whose keys and values correspond to the original and pretty
        labels, respectively.

    Returns
    -------
    pretty_label: str or a list of str
        Converted labels.
    """
    label = [label] if isinstance(label, str) else label
    pretty_label = [None] * len(label)

    for i, p in enumerate(label):
        if not isinstance(p, str):
            raise TypeError("`label` must be a string or a list of strings.")
        pretty_label[i] = pretty_dict[p] if p in pretty_dict.keys() else p

    return pretty_label if len(pretty_label) > 1 else pretty_label[0]


def parse_features(features):
    """
    Parse a string of features looking like `x,y,z`.

    Parameters
    ----------
    features : str
        Features string.

    Returns
    -------
    features : list of str
        Unpacked features.
    """
    if not isinstance(features, str):
        raise TypeError("`features` must be a string, e.g. `'gbar,SBdisk'`.")
    if " " in features:
        raise ValueError("`features` string cannot contain whitespace!")

    features = features.split(",")
    return features


###############################################################################
#                        Feature LaTeX names                                  #
###############################################################################


names = {
    "gobs": r"$g_{\rm obs}$",
    "e_gobs": r"$\Delta g_{\rm obs} / g_{\rm obs}$",
    "gbar": r"$g_{\rm bar}$",
    "e_gbar": r"$\Delta g_{\rm bar} / g_{\rm bar}$",
    "Vobs": r"$V_{\rm obs}$",
    "Vbar": r"$V_{\rm bar}$",
    "e_Vobs": r"$\Delta V_{\rm obs} / V_{\rm obs}$",
    "r": r"$r / R_{\rm eff}$",
    "SBdisk": r"$\Sigma_{\rm disk}$",
    "SBbul": r"$\Sigma_{\rm bul}$",
    "SB": r"$\Sigma_{\rm tot}$",
    "dist": r"$D$",
    "e_dist": r"$\Delta D$",
    "inc": r"$i$",
    "e_inc": r"$\Delta i$",
    "L36": r"$L_{3.6}$",
    "MHI": r"$M_{\rm HI}$",
    "Reff": r"$R_{\rm eff}$",
    "rand_rc": r"$\mathcal{N}_{\rm RC}$",
    "rand_gal": r"$\mathcal{N}_{\rm gal}$",
    "type": r"$T$",
    "log_gNext": r"$\log g_{\rm N, ext}$",
    "log_gext": r"$\log g_{\rm ext}$",
    "log_eN_noclust": r"$e_{\rm N}$",
    "log_eN_maxclust": r"$e_{\rm N}$",
    }

names_wunits = {
    "gobs": r"$\log g_{\rm obs} / (\mathrm{m}\mathrm{s}^{-2})$",
    "e_gobs": r"$\Delta g_{\rm obs} / g_{\rm obs}$",
    "gbar": r"$\log g_{\rm bar} / (\mathrm{m}\mathrm{s}^{-2})$",
    "e_gbar": r"$\Delta g_{\rm bar} / g_{\rm bar}$",
    "Vobs": r"$V_{\rm obs} / (\mathrm{km} \mathrm{s}^{-1})$",
    "e_Vobs": r"$\Delta V_{\rm obs} / V_{\rm obs}$",
    "r": r"$\log r / R_{\rm eff}$",
    "SBdisk": r"$\log \Sigma_{\rm disk}$",
    "SBbul": r"$\log \Sigma_{\rm bul}$",
    "dist": r"$D$",
    "e_dist": r"$\Delta D$",
    "inc": r"$i~[\mathrm{degrees}]$",
    "e_inc": r"$\Delta i~[\mathrm{degrees}]$",
    "L36": r"$\log L_{3.6}$",
    "MHI": r"$\log M_{\rm HI}$",
    "Reff": r"$\log R_{\rm eff}$",
    }
