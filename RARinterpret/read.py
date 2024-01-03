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
Data processing functions and a file system.
"""
from os.path import join
from warnings import filterwarnings, catch_warnings

import numpy
from astropy import units

from .analytical import MockRAR, MockSBJobs

KPC2KM = units.kpc.to(units.km)

###############################################################################
#                         The RAR data frame                                  #
###############################################################################


class RARFrame:
    """
    Data frame-like object for convenient I/O and data pre-processing.

    Parameters
    ----------
    rc_fpath : str, optional
        Path to the text file containing the RC information.
        By default `../data/RAR.dat`.
    lookup_fpath : str, optional
        Path to the text file containing the galaxy look-up information.
        By default `../data/Name_lookup.dat`.
    veldecomp_fpath : str, optional
        Path to the text file containing the velocity decomposition data. By
        default `../data/MassModels_Lelli2016c.txt`.
    efep1_fpath : str, optional
        Path to the EFE paper 1 data.
        By default `../data/SPARC_EFE_final_400.dat`.
    efep2_fpath : str, optional
        Path to the EFE paper 2 data. By default `../data/eN_Final.dat`.
    """
    _data = None
    _names = None

    def __init__(self, rc_fpath="../data/RAR.dat",
                 lookup_fpath="../data/Name_lookup.dat",
                 veldecomp_fpath="../data/MassModels_Lelli2016c.txt",
                 sparclelli2016c_fpath="../data/SPARC_Lelli2016c.txt",
                 efep1_fpath="../data/SPARC_EFE_final_400.dat",
                 efep2_fpath="../data/eN_Final.dat"):
        rc_data = self.read_rc(rc_fpath)
        lookup_data, self._names = self.read_galaxy_lookup(lookup_fpath)
        self._data = self.add_fromlookup(rc_data, lookup_data)

        # Get a mapping from a galaxy naem to an index
        self._name2gindx = {}
        for name, i in zip(self._names, lookup_data["index"]):
            self._name2gindx.update({name: i})

        # Add the velocity decomposition
        veldecomp = self.read_velocity_decomposition(veldecomp_fpath)
        self._data = self.add_velocity_decomposition(*veldecomp)
        # Add e_L36
        lelli2016 = self.read_sparclelli2016c(sparclelli2016c_fpath)
        self._data = self.add_sparclelli2016(*lelli2016)
        # Add the EFE
        efp1 = self.read_efpaper1(efep1_fpath)
        self._data = self.add_fromefe(*efp1)
        efp2 = self.read_efpaper2(efep2_fpath)
        self._data = self.add_fromefe(*efp2)

        # Add Vbar, SB and Jobs
        vals = [self.Vbar, self["SBdisk"] + self["SBbul"],
                self["Vobs"]**3 / self["r"]**2]
        names = ["Vbar", "SB", "Jobs"]
        self._data = add_columns(self._data, vals, names)

    @property
    def data(self):
        """
        The data array. Contains the RC samples with their associated galaxy
        information.

        Returns
        -------
        data : structured array
        """
        return self._data

    @property
    def size(self):
        """
        Number of RC samples.

        Returns
        -------
        size : int
        """
        return self.data.size

    @property
    def normalised_data(self):
        """
        The normalised data array.

        Returns
        -------
        norm_data : structured array
            The normalised array.
        """
        data = numpy.copy(self.data)
        # Replace some 0s
        data = replace_values(data, "SBdisk", 0, 0.001)
        data = replace_values(data, "SBbul", 0, 0.001)
        data = replace_values(data, "SB", 0, 0.001)
        # Get r in units of Reff
        data["r"] /= data["Reff"]
        # Log-transform a selected columns
        log_cols = ["gbar", "gobs", "r", "MHI", "L36",
                    "SBdisk", "SBbul", "Reff", "Vbar", "Vobs", "SB", "Jobs"]
        data = log_columns(data, log_cols)
        return data

    @property
    def names(self):
        """
        The galaxy names.

        Returns
        -------
        names : 1-dimensional array
            Array of names.
        """
        return self._names

    @property
    def keys(self):
        """
        Data keys.

        Returns
        -------
        keys : 1-dimensional array of str
        """
        return self._data.dtype.names

    @staticmethod
    def read_rc(fpath):
        """
        Read in the RAR rotation curve (RC) data. Must be in a specific format.
        Removes `Vobs` and `e_Vobs`.

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            RAR data.
        """
        data = numpy.loadtxt(fpath)
        # An awkward sequence of operations to extract the columns
        with open(fpath, "r") as file:
            cols = file.readline().lstrip("# ").rstrip("/n")
            cols = cols.split(", ")
            cols = [col.split(" [")[0] for col in cols]
        # Preallocate an array
        dtype = {"names": cols,
                 "formats": [numpy.int64] + [numpy.float64] * (len(cols) - 1)}

        # Catch a warning that occus when converting NaN to int
        with catch_warnings():
            filterwarnings(
                "ignore", message=".*invalid value encountered in cast.*",
                category=RuntimeWarning
                )
            out = numpy.full(data.shape[0], numpy.nan, dtype=dtype)
        # Fill the array
        for i, col in enumerate(cols):
            out[col] = data[:, i]
        return out

    @staticmethod
    def read_galaxy_lookup(fpath):
        """
        Read in the RAR galaxy look-up data. Must be in a specific format.

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            The galaxy look-up data.
        names : 1-dimensional array
            Array of galaxy names.
        """
        # names_col containts strings, so we extract it separately
        names_col = 1
        cols = [i for i in range(10) if i != names_col]
        # Open arrays
        data = numpy.genfromtxt(fpath, usecols=cols)
        names = numpy.genfromtxt(fpath, dtype=str, usecols=names_col)
        # Get the column names
        with open(fpath, "r") as file:
            cols = file.readline().lstrip("# ").rstrip("\n").split(" ")
            cols = [col for i, col in enumerate(cols) if i != names_col]

        # Preallocate an array
        dtype = {"names": cols,
                 "formats": [numpy.int64] + [numpy.float64] * (len(cols) - 1)}
        # Catch a warning that occus when converting NaN to int
        with catch_warnings():
            filterwarnings(
                "ignore", message=".*invalid value encountered in cast.*",
                category=RuntimeWarning
                )
            out = numpy.full(data.shape[0], numpy.nan, dtype=dtype)
        # Fill the array
        for i, col in enumerate(cols):
            out[col] = data[:, i]
        return out, names

    @staticmethod
    def read_velocity_decomposition(fpath):
        """
        Read in the velocity decomposition data. Must be in a spefic format
        matching [1].

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            The external field data
        names : 1-dimensional array
            Array of galaxy names.

        References
        ----------
        [1] http://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt
        """
        data = numpy.genfromtxt(fpath, skip_header=25)[:, 1:]
        names = numpy.genfromtxt(fpath, skip_header=25, usecols=0, dtype=str)
        return data, names

    @staticmethod
    def read_sparclelli2016c(fpath):
        """
        Read in the SPARC galaxy data. Must be in a spefic format matching [1].

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            The external field data
        names : 1-dimensional array
            Array of galaxy names.

        References
        ----------
        [1] http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt
        """
        data = numpy.genfromtxt(fpath, skip_header=98)[:, 1:]
        names = numpy.genfromtxt(fpath, skip_header=98, usecols=0, dtype=str)
        return data, names

    @staticmethod
    def read_efpaper1(fpath):
        """
        Read in the external field data of paper 1. Must be in a spefic format.

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            The external field data
        names : 1-dimensional array
            Array of galaxy names.
        """
        data = numpy.genfromtxt(fpath, usecols=[1, 2], delimiter=" ")
        names = numpy.genfromtxt(fpath, usecols=0, dtype=str)
        # Pre-allocate an array
        cols = ["log_gNext", "log_gext"]
        dtype = {"names": cols, "formats": [numpy.float64] * 2}

        out = numpy.full(data.shape[0], numpy.nan, dtype=dtype)
        for i, col in enumerate(cols):
            out[col] = data[:, i]

        return out, names

    @staticmethod
    def read_efpaper2(fpath):
        """
        Read in the external field data of paper 2. Must be in a spefic format.

        Parameters
        ----------
        fpath : str
            The file path.

        Returns
        -------
        data : structured array
            The external field data
        names : 1-dimensional array
            Array of galaxy names.
        """
        data = numpy.genfromtxt(fpath, usecols=[1, 2, 3, 4], delimiter=" ")
        names = numpy.genfromtxt(fpath, usecols=0, dtype=str)
        # Pre-allocate an array
        cols = ["log_eN_noclust", "e_log_eN_noclust",
                "log_eN_maxclust", "e_log_eN_maxclust"]
        dtype = {"names": cols, "formats": [numpy.float64] * 4}

        out = numpy.full(data.shape[0], numpy.nan, dtype=dtype)
        for i, col in enumerate(cols):
            out[col] = data[:, i]
        return out, names

    @staticmethod
    def add_fromlookup(data_rc, data_gal):
        """
        Add rows from the galaxy lookup array `data_gal` to the rotation curve
        array `data_rc`. In both arrays their linking `index` must be the 0th
        column.

        Parameters
        ----------
        data_rc : structured array
            The RC array.
        data_gal : structured array
            The galaxy look-up array.

        Returns
        -------
        out : structured array
            The RC array with the matched galaxy look-up data.
        """
        # Get the individual dtypes
        descr_rc = data_rc.dtype.descr
        descr_gal = data_gal.dtype.descr
        # Pop the "index" dtype from the gal descr
        descr_gal.pop(data_gal.dtype.names.index("index"))

        # Preallocate an array
        with catch_warnings():
            filterwarnings(
                "ignore", message=".*invalid value encountered in cast.*",
                category=RuntimeWarning
                )
            out = numpy.full(data_rc.size, numpy.nan,
                             dtype=descr_rc + descr_gal)
        # Fill the array with RC data
        for par in data_rc.dtype.names:
            out[par] = data_rc[par]
        # Fill the look-up data at the corresponding indices
        pars = [descr[0] for descr in descr_gal]
        for k in data_gal["index"]:
            mask = data_rc["index"] == k
            for par in pars:
                out[par][mask] = data_gal[par][k]
        return out

    def add_velocity_decomposition(self, data, names):
        """
        Add the velocity decomposition data.

        Parameters
        ----------
        data : 2-dimensional array of shape `(n_samples, n_features)`
            Velocity decomposition data. Must contain `r`, `Vgas`, `Vdisk` and
            `Vbul` at zero-indexed positions 1, 4, 5, 6, respectively.
        names : 1-dimensional array of shape (`n_samples,`)
            Array of galaxy names corresponding to `data`.

        Returns
        -------
        out : structured array
            Data with appended values.
        """
        # Pre-allocate the output array
        cols = ["Vgas", "Vdisk", "Vbul"]
        out = numpy.full((self.size, 3), numpy.nan)
        # Loop over the frame's galaxies
        for name in self.names:
            # Get the properties of this galaxy from the decomposition file
            r, Vgas, Vdisk, Vbul = [data[names == name][:, i]
                                    for i in (1, 4, 5, 6)]

            # Frame indices of this galaxy
            ks = numpy.where(self["index"] == self._name2gindx[name])[0]

            for k in ks:
                # Find which frame position matches this galaxy and decomp file
                match = self["r"][k] == r
                assert numpy.sum(match) == 1
                out[k, 0] = Vgas[match]
                out[k, 1] = Vdisk[match]
                out[k, 2] = Vbul[match]

        self._data = add_columns(self._data, out, cols)
        return self._data

    def add_sparclelli2016(self, data, names):
        """
        Add `e_L36`, `SBdisk0` and `Rdisk `from the SPARC data.

        Parameters
        ----------
        data : 2-dimensional array of shape `(n_samples, n_features)`
            Velocity decomposition data. Assumes `e_L36` to be contained at
            the 6th (zero-indexed) position.
        names : 1-dimensional array of shape (`n_samples,`)
            Array of galaxy names corresponding to `data`.

        Returns
        -------
        out : structured array
            Data with appended values.
        """
        cols = [(7, "e_L36"), (10, "Rdisk0"), (11, "SBdisk0")]
        for k, col in cols:
            x = numpy.full(len(self), numpy.nan)
            # Loop over the frame's galaxies
            for name in self.names:
                value = data[names == name, k]
                assert value.size == 1
                x[self["index"] == self._name2gindx[name]] = value
            self._data = add_columns(self._data, x, col)
        return self._data

    def add_fromefe(self, data_efe, names_efe):
        """
        Add the EFE paper (either 1 or 2) data to the RC array. If EFE data
        not available for a galaxy then its RC samples are removed.

        Parameters
        ----------
        data_efe : structured array
            Array from `self.read_efpaper1`
        names_efe : 1-dimensional str array
            Array of galaxy names corresponding to rows of `data_efe`.

        Returns
        -------
        data : structured array
            Data with appended values.
        """
        nvals = len(data_efe.dtype.names)
        data_append = numpy.full((self.data.size, nvals), numpy.nan)

        for i, name in enumerate(names_efe):
            try:
                k = self._name2gindx[name]
                for j, par in enumerate(data_efe.dtype.names):
                    data_append[self["index"] == k, j] = data_efe[par][i]
            except KeyError:
                pass

        self._data = add_columns(self._data, data_append, data_efe.dtype.names)
        return self._data

    def read_urar(self, dpath=None, lpath=None):
        """
        Read RAR-transformed data from Harry's files.

        Parameters
        ----------
        dpath : str
            Path to the data file.
        lpath : str
            Path to the labels file.

        Returns
        -------
        out : structured array
            Data array with keys `r`, `gbar`, `gobs`, `Vbar`, `Vobs`.
        """
        folder = "/mnt/zfsusers/hdesmond/Big_Inference/"
        if dpath is None:
            dpath = join(folder, "samples_52_scatter_EFE2.npy")
        if lpath is None:
            lpath = join(folder, "labels_52_scatter_EFE2.npy")

        # Load data and labels. Make mapping from labels to column number
        data = numpy.load(dpath)
        labels = numpy.load(lpath)
        lab2col = {label: i for i, label in enumerate(labels)}

        # Preallocate output arrays
        nsamples = self["index"].size
        dtype = {"names": ["r", "gbar", "gobs", "Vbar", "Vobs"],
                 "formats": [numpy.float64] * 5}
        out = numpy.full(nsamples, numpy.nan, dtype=dtype)

        # ML bulge are stored as 0, ..., 30 but that does not correspond
        # to galaxy indices, only in their ordering. Correct like this.
        arr2bulind = [i for i in range(self["index"].max())
                      if numpy.any(self["Vbul"][self["index"] == i] > 0)]
        arr2bulind = {i: j for j, i in enumerate(arr2bulind)}

        for n in numpy.unique(self["index"]):
            m = self["index"] == n     # Mask of this RC
            k0 = numpy.where(m)[0][0]  # First array index of this RC

            try:
                kk = arr2bulind[n]
                mlbul = numpy.median(data[:, lab2col["ML_bul_" + str(kk)]])
            except KeyError:
                mlbul = 0.

            mldisk = numpy.median(data[:, lab2col["ML_disk_" + str(n)]])
            mlgas = numpy.median(data[:, lab2col["ML_gas_" + str(n)]])
            # Scale M/L of the bulge and disk by luminosity
            L36 = numpy.median(data[:, lab2col["L36_" + str(n)]]) * 1e-9
            mlbul *= L36 / self["L36"][k0]
            mldisk *= L36 / self["L36"][k0]

            # Calculate Vbar^2 from components
            Vbar2 = (mldisk * self["Vdisk"][m] * numpy.abs(self["Vdisk"][m])
                     + mlbul * self["Vbul"][m] * numpy.abs(self["Vbul"][m])
                     + mlgas * self["Vgas"][m] * numpy.abs(self["Vgas"][m]))
            # Calculate gbar. Independent of distance!
            out["gbar"][m] = Vbar2 / self["r"][m] * 1e13 / KPC2KM

            # Now rescale gobs
            out["gobs"][m] = self["gobs"][m]
            # Scale it by distance
            dist = numpy.median(data[:, lab2col["Dist_" + str(n)]])
            out["gobs"][m] *= self["dist"][k0] / dist
            # Scale it by inclination
            inc = numpy.deg2rad(
                numpy.median(data[:, lab2col["Inc_" + str(n)]]))
            out["gobs"][m] *= (numpy.sin(numpy.deg2rad(self["inc"][k0]))
                               / numpy.sin(inc))**2

            # Lastly also scale the radius
            out["r"][m] = self["r"][m] * dist / self["dist"][k0]

        # Now calculate Vbar and Vobs from the extracted values
        out["Vbar"] = (out["gbar"] * out["r"] * 1e-13 * KPC2KM)**0.5
        out["Vobs"] = (out["gobs"] * out["r"] * 1e-13 * KPC2KM)**0.5

        return out

    def read_urar_single(self, galaxy_name, take_every=1,
                         dpath=None, lpath=None):
        """
        Generate the RAR-transformed data for a single galaxy.

        Parameters
        ----------
        galaxy_name : str
            Name of the galaxy.
        take_every : int, optional
            Take every `take_every` sample from the chain.
        dpath : str
            Path to the data file.
        lpath : str
            Path to the labels file.

        Returns
        -------
        data : dict
            Dictionary with keys `gbar`, `gobs` and `r`.
        """
        if galaxy_name not in self._name2gindx:
            raise KeyError(f"Galaxy `{galaxy_name}` not found in the frame.")
        if not isinstance(take_every, int) or take_every < 1:
            raise ValueError("`take_every` must be a positive integer.")

        gindex = self._name2gindx[galaxy_name]
        k0 = numpy.where(self["index"] == gindex)[0][0]
        m = self["index"] == gindex     # Mask of this RC

        folder = "/mnt/zfsusers/hdesmond/Big_Inference/"
        if dpath is None:
            dpath = join(folder, "samples_52_scatter_EFE2.npy")
        if lpath is None:
            lpath = join(folder, "labels_52_scatter_EFE2.npy")

        # ML bulge are stored as 0, ..., 30 but that does not correspond
        # to galaxy indices, only in their ordering. Correct like this.
        arr2bulind = [i for i in range(self["index"].max())
                      if numpy.any(self["Vbul"][self["index"] == i] > 0)]
        arr2bulind = {i: j for j, i in enumerate(arr2bulind)}

        # Load the data and labels for the entire SPARC data set.
        # `data` is a 2-dimensional array of shape (nchain_samples, nlabels)
        data = numpy.load(dpath)
        labels = numpy.load(lpath)

        # Create the mapping from labels to columns numbers.
        lab2col = {label: i for i, label in enumerate(labels)}
        # Take every `take_every` sample from the chain.
        data = data[::take_every, :]

        # Get the samples of M/L, L36, distance and inclination of the galaxy
        try:
            mlbul = data[:, lab2col[f"ML_bul_{arr2bulind[gindex]}"]]
            mlbul = mlbul.reshape(-1, 1)
        except KeyError:
            mlbul = 0.
        mldisk = data[:, lab2col[f"ML_disk_{gindex}"]].reshape(-1, 1)
        mlgas = data[:, lab2col[f"ML_gas_{gindex}"]].reshape(-1, 1)
        L36 = data[:, lab2col[f"L36_{gindex}"]].reshape(-1, 1) * 1e-9
        inc = numpy.deg2rad(data[:, lab2col[f"Inc_{gindex}"]].reshape(-1, 1))
        dist = data[:, lab2col[f"Dist_{gindex}"]].reshape(-1, 1)

        a0 = data[:, lab2col["a0"]].reshape(-1, 1)
        scatter = data[:, lab2col["scatter"]].reshape(-1, 1)

        # Calculate gbar = Vbar^2 / r from components. It is independent of
        # of the distance and only depends on the M/L and L36
        gbar = (
            L36 / self["L36"][k0] * mldisk * self["Vdisk"][m] * numpy.abs(self["Vdisk"][m]) # noqa
            + L36 / self["L36"][k0] * mlbul * self["Vbul"][m] * numpy.abs(self["Vbul"][m])  # noqa
            + mlgas * self["Vgas"][m] * numpy.abs(self["Vgas"][m]))
        gbar /= self["r"][m]
        gbar *= 1e13 / KPC2KM

        # Now rescale gobs
        gobs = self["gobs"][m] * (self["dist"][k0] / dist)
        inc0 = numpy.deg2rad(self["inc"][k0])

        gobs *= (numpy.sin(inc0) / numpy.sin(inc))**2

        # Lastly also scale the radius
        rad = self["r"][m] * dist / self["dist"][k0]

        return {"gbar": gbar,
                "gobs": gobs,
                "r": rad,
                "ML_disk": mldisk,
                "ML_bul": mlbul,
                "ML_gas": mlgas,
                "L36": L36,
                "inc": inc,
                "dist": dist,
                "a0": a0,
                "scatter": scatter,
                }

    def generate_log_variance(self, feat):
        r"""
        Return approximate Gaussian-propagated variance of a logarithm of a
        feature.

        Parameters
        ----------
        feat : str or len-2 tuple
            Feature whose variance to be obtained. If a tuple assumed to be
            `(alpha, beta)`.

        Returns
        -------
        var : float
        """
        if isinstance(feat, (tuple, list)):
            assert len(feat) == 2
            alpha, beta = feat  # Unpack `feat` and rewrite it with a str
            feat = "gencomb"

        # Convert inclination to radians
        if feat in ["Vobs", "Jobs", "gencomb"]:
            inc = numpy.deg2rad(self["inc"])
            e_inc = numpy.deg2rad(self["e_inc"])
        else:
            inc, e_inc = None, None
        # If a gencomb then alpha, beta must be given
        if feat == "gencomb":
            assert alpha is not None and beta is not None

        if feat == "Vobs":
            relerr = numpy.sqrt(
                (self["e_Vobs"] / self["Vobs"])**2
                + (e_inc / numpy.tan(inc))**2)
        elif feat == "Jobs":
            relerr = numpy.sqrt(
                (3 * self["e_Vobs"] / self["Vobs"])**2
                + (3 * e_inc / numpy.tan(inc))**2
                + (2 * self["e_dist"] / self["dist"]**2))
        elif feat == "gencomb":
            relerr = numpy.sqrt(
                (alpha * self["e_Vobs"] / self["Vobs"])**2
                + (alpha * e_inc / numpy.tan(inc))**2
                + (beta * self["e_dist"] / self["dist"])**2)
        elif feat == "Vbar":
            relerr = 0.5 * numpy.sqrt(
                (self["e_gbar"] / self["gbar"])**2
                + (self["e_dist"] / self["dist"])**2)
        elif feat == "SB":
            relerr = self["e_L36"] / self["L36"]
        elif "e_{}".format(feat) in self.keys:
            relerr = self["e_{}".format(feat)] / self[feat]
        else:
            relerr = numpy.zeros_like(self["e_gbar"])

        return (relerr / numpy.log(10))**2

    def make_Xy(self, target, features=None, append_variances=False,
                gen_model=None, thetas=None, seed=None, remove_efe_nan=False,
                mock_kind="RAR", dtype=numpy.float64):
        r"""
        Make feature and target arrays for predicting `target`.

        Parameters
        ----------
        target : str or a len-2 tuple
            Target variable name. If a tuple assumed to be `(alpha, beta)`.
        features : str or a list of str, optional.
            The feature names. By default all columns are returned.
        append_variances : bool, optional
            Whether to append variances of the first feature and the target
            to `y`.
        gen_model : Instance of py:class`BaseRAR`
            Model used to generate new mocks. If provided then this function
            behaves as a mock generator, however, the target must be `gobs`
            or `gencomb`.
        thetas : 1-dimensional array, optional
            Parameters of `gen_model`.
        seed : int
            Random seed.
        remove_efe_nan : bool, optional
            Whether to remove galaxies with no EFE parameters.
        mock_kind : str, optional
            Kind of mocks to generate. Must be either `RAR` or `SB-Jobs`.
        dtype : dtype
            Output array dtype.

        Returns
        -------
        X : 2-dimensional array
            The feature array of shape `(n_samples, n_features)`.
        y : 1- or 2-dimensional array
            The target array. If `append_variances` is `True`, then the output
            shape is `(n_samples, 3)`, where the colums are the `gobs`,
            variance of `gobs` and varince of `gbar`.
        features : list of str
            Feature list corresponding to `X`.
        """
        assert mock_kind in ["RAR", "SB-Jobs"]
        # Check the target
        if isinstance(target, (tuple, list)):
            assert len(target) == 2
            alpha, beta = target  # Unpack `target` and rewrite it with a str
        else:
            alpha, beta = None, None

        data = self.normalised_data
        features = [features] if isinstance(features, str) else features
        # Otherwise take all features but remove these.
        if features is None:
            features = list(data.dtype.names)
            for p in ("index", "gobs", "e_gobs", "Vobs", "e_Vobs"):
                if p in features:
                    features.remove(p)

        # Generate mock data. Just replace values in `data`
        if gen_model is not None:
            assert target == "gobs" or isinstance(target, (tuple, list))
            # If generating a general combination make sure these are given
            if target == "gencomb":
                assert alpha is not None and beta is not None

            if mock_kind == "RAR":
                gen_data = MockRAR()
            else:
                gen_data = MockSBJobs()
            thetas = gen_model.x0 if thetas is None else thetas
            pars = ("index", "Vgas", "Vdisk", "Vbul", "r", "dist",
                    "e_dist", "inc", "e_inc", "L36", "e_L36",
                    "SBdisk", "SBbul", "Vobs", "e_Vobs",
                    "log_eN_noclust", "e_log_eN_noclust")
            Vbar, gbar, L36, SBdisk, SBbul, SB, r, y = gen_data.rvs(
                gen_model, thetas, *[self[p] for p in pars],
                return_log=True, alpha=alpha, beta=beta, seed=seed)
            # Replace all the appropriate values
            data["Vbar"] = Vbar
            data["gbar"] = gbar
            data["L36"] = L36
            data["SBdisk"] = SBdisk
            data["SBbul"] = SBbul
            data["SB"] = SB
            data["r"] = r - numpy.log10(self["Reff"])
        else:
            if isinstance(target, str):
                y = data[target]
            else:
                y = self["Vobs"]**alpha / self["r"]**beta * 1e13 / KPC2KM
                y = numpy.log10(y)

        # Pre-allocate and fill the feature array
        X = numpy.full((data.size, len(features)), numpy.nan, dtype=dtype)
        for i, feature in enumerate(features):
            X[:, i] = data[feature]

        if append_variances:
            y = numpy.vstack([
                y, self.generate_log_variance(features[0]),
                self.generate_log_variance(target)]).T
        y = y.astype(dtype)  # Enforce dtype

        # Remove no EFE galaxies
        if remove_efe_nan:
            mask = ~numpy.isnan(self["log_eN_noclust"])
            X, y = X[mask, :], y[mask]

        return X, y, features

    def shuffle_galaxy(self, x):
        """
        Shuffle a galaxy-wide property of RC samples. Assumes that items of
        `x` link to galaxy indices according to `self["index"]`. Treats values
        of `numpy.nan` as any other number and assigns them randomly to
        another galaxy.

        Parameters
        ----------
        x : 1-dimensional array of shape `(self.size, )`

        Returns
        -------
        out : 1-dimensional array of shape `(self.size, )`
        """
        if not x.ndim == 1 and x.size != self.size:
            raise TypeError("`x` must be an array of shape `(self.size, )`.")

        # Unique indices of galaxies and their properties
        gal_indxs = numpy.unique(self["index"])
        gal_x = numpy.full(gal_indxs.size, numpy.nan)
        for i, j in enumerate(gal_indxs):
            subx = x[self["index"] == j]
            # Check that all extracted properties of a gal are equal
            if not numpy.array_equal(subx, subx, equal_nan=True):
                raise ValueError("Properties of galaxies at a single index "
                                 "not equal! Is this a galaxy-wide prop?")
            # Assign it to the vector
            gal_x[i] = subx[0]

        # Now shuffle gal_x and assign it back to x
        numpy.random.shuffle(gal_x)
        out = numpy.full_like(x, numpy.nan)
        for i, j in enumerate(gal_indxs):
            out[self["index"] == j] = gal_x[i]

        return out

    @property
    def Vbar(self):
        r"""
        The baryonic rotational velocity. Explicitly calculated from `gbar`
        and `r`.

        Returns
        -------
        Vbar : 1-dimensional array
            Vbar in units of :math:`\mathrm{km} / \mathrm{s}`.
        """
        return numpy.sqrt(1e-10 * self["gbar"] * self["r"] * KPC2KM) * 1e-3

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return self.data.size


###############################################################################
#                   Structured array handling tools                           #
###############################################################################


def replace_values(data, col, original_value, new_value):
    """
    Replace values `original_value` of column `col` with `new_value`. Modifies
    the original `data` array.

    Parameters
    ----------
    data : structured array
        The data array.
    col : str
        The column where to replace values.
    original_value : float or any other type identifiable with `==`
        The value to be replaced.
    new_value : same as `original_value`
        The new value.

    Returns
    -------
    data : 2-dimensional array
        The modified data array.
    """
    mask = data[col] == original_value
    data[col][mask] = new_value
    return data


def log_columns(data, cols):
    r"""
    Take :math:`log_{10}` of columns `cols`. Modifies the original array.

    Parameters
    ----------
    data : structured array
        The data array.
    cols : list of str
        Columns to be taken a log of.

    Returns
    -------
    data : structured array
        The transformed data array.
    """
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        data[col] = numpy.log10(data[col])
    return data


def add_columns(arr, X, cols):
    """
    Add new columns to a record array `arr`. Creates a new array.

    Parameters
    ----------
    arr : record array
        The record array to add columns to.
    X : (list of) 1-dimensional array(s) or 2-dimensional array
        Columns to be added.
    cols : str or list of str
        Column names to be added.

    Returns
    -------
    out : record array
        The new record array with added values.
    """
    # Make sure cols is a list of str and X a 2D array
    cols = [cols] if isinstance(cols, str) else cols
    if isinstance(X, numpy.ndarray) and X.ndim == 1:
        X = X.reshape(-1, 1)
    if isinstance(X, list) and all(x.ndim == 1 for x in X):
        X = numpy.vstack([X]).T
    if len(cols) != X.shape[1]:
        raise ValueError("Number of columns of `X` does not match `cols`.")
    if arr.size != X.shape[0]:
        raise ValueError("Number of rows of `X` does not match size of `arr`.")

    # Get the new data types
    dtype = arr.dtype.descr
    for i, col in enumerate(cols):
        dtype.append((col, X[i, :].dtype.descr[0][1]))

    # Fill in the old array
    with catch_warnings():
        filterwarnings(
            "ignore", message=".*invalid value encountered in cast.*",
            category=RuntimeWarning
            )
        out = numpy.full(arr.size, numpy.nan, dtype=dtype)
    for col in arr.dtype.names:
        out[col] = arr[col]
    for i, col in enumerate(cols):
        out[col] = X[:, i]

    return out


def rm_columns(arr, cols):
    """
    Remove columns `cols` from a record array `arr`. Creates a new array.
    Parameters
    ----------
    arr : record array
        The record array to remove columns from.
    cols : str or list of str
        Column names to be removed.
    Returns
    -------
    out : record array
        Record array with removed columns.
    """
    # Check columns we wish to delete are in the array
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        if col not in arr.dtype.names:
            raise ValueError("Column `{}` not in `arr`.".format(col))

    # Get a new dtype without the cols to be deleted
    new_dtype = []
    for dtype, name in zip(arr.dtype.descr, arr.dtype.names):
        if name not in cols:
            new_dtype.append(dtype)

    # Allocate a new array and fill it in.
    out = numpy.full(arr.size, numpy.nan, new_dtype)
    for name in out.dtype.names:
        out[name] = arr[name]

    return out


def extract_col_from_X(X, col, features):
    """
    Extract a column from a 2-dimensional array whose columns are features `X`.

    Parameters
    ----------
    X : 2-dimensional array
        Feature array.
    col : str
        Column to be extracted.
    features : list of str
        Column names of `X`.

    Returns
    -------
    X : 2-dimensional array
        Feature array after extracting `col`.
    z : 1-dimensional array
        Array containing `col`.
    features : list of str
        Features after removing `col`.
    """
    # Extract gbar from X and remove it from features
    z = X[:, features.index(col)]
    X = numpy.delete(X, features.index(col), axis=1)
    features.remove(col)
    return X, z, features
