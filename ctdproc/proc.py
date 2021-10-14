#!/usr/bin/env python
# coding: utf-8
import gsw
import matplotlib as mpl
import numpy as np
import xarray as xr
from scipy import fft, optimize, signal, stats

from . import calcs, helpers

# We are running into trouble if the default backend is not installed
# in the current python environment. Try to import pyplot the default
# way first. If this fails, set backend to agg which should always work.
try:
    import matplotlib.pyplot as plt
except ImportError:
    mpl.use("agg")
    import matplotlib.pyplot as plt


def run_all(ds):
    """
    Run all standard processing steps on raw CTD time series.

    Parameters
    ----------
    data : xarray.Dataset
            CTD time series data structure

    Returns
    -------
    datad : xarray.Dataset
            Downcast time series
    datau : xarray.Dataset
            Upcast time series
    """
    ds = cleanup(ds)

    ds_updown = split_updn(ds)

    for v, d in ds_updown.items():
        add_tcfit_default(d)
        d = d.dropna("time", how="any")
        d = phase_correct(d)
        d = calcs.swcalcs(d)
        d = rmloops(d)
        d = cleanup_ud(d)
        ds_updown[v] = d

    return ds_updown


def plot_profile(ds, var_list):
    for i, v in enumerate(var_list, start=1):
        plt.subplot(1, len(var_list), i)
        if "time" in ds.dims:
            ds[v].plot(x="time")
        elif "depth" in ds.dims:
            ds[v].plot(y="depth")


def plot_TS(ds):
    plt.plot(ds["t1"], ds["s1"], alpha=0.1)
    plt.plot(ds["t2"], ds["s2"], alpha=0.1)


def add_tcfit_default(ds):
    """
    Get default values for tc fit range depending on depth of cast.

    Range for tc fit is 200dbar to maximum pressure if the cast is
    shallower than 1000dbar, 500dbar to max pressure otherwise.

    Parameters
    ----------
    ds : xarray.Dataset
            CTD time series data structure

    Returns
    -------
    tcfit : tuple
            Upper and lower limit for tc fit in phase_correct.
    """
    if ds.p.max() > 1000:
        tcfit = [500, ds.p.max().data]
    elif ds.p.max() > 300:
        tcfit = [200, ds.p.max().data]
    else:
        tcfit = [50, ds.p.max().data]
    ds.attrs["tcfit"] = tcfit


def cleanup(ds):
    """
    Clean up CTD raw time series.

    - despike pressure
    - eliminate data near surface
    - remove spikes in other data
    - remove smaller T, C, glitches
    """

    # despike pressure
    ds["p"].data = helpers.glitchcorrect(
        ds.p.data, ds.attrs["diff_p"], ds.attrs["prod_p"]
    )
    ipmax = np.argmax(ds.p.data)

    # eliminate near-surface data
    ptop = 1
    fdeep = np.squeeze(np.where(ds.p.data > ptop))
    ideepstart, ideepstop, ideeplen = helpers.findsegments(fdeep)
    ii = np.max(ideepstart[ideepstart < ipmax])
    jj = np.min(ideepstop[ideepstop >= ipmax])
    ds = ds.isel(time=range(ii, jj + 1))

    # remove spikes in temperature
    ib = np.squeeze(
        np.where(np.absolute(np.diff(ds.t1.data)) > ds.attrs["spike_thresh_t"])
    )
    ds.t1[ib] = np.nan
    ib = np.squeeze(
        np.where(np.absolute(np.diff(ds.t2.data)) > ds.attrs["spike_thresh_t"])
    )
    ds.t2[ib] = np.nan

    # remove out of range values
    ds = preen_ctd(ds)
    # no trans, fl ***

    # remove nans at start
    fnan1 = np.squeeze(np.where(np.isnan(ds.c1.data)))
    fnan2 = np.squeeze(np.where(np.isnan(ds.c2.data)))
    if np.array_equiv(fnan1, fnan2) is False:
        print("warning: NaNs index different in data.c1 and data.c2")
    if fnan1.size > 0:
        istart, istop, ilen = helpers.findsegments(fnan1)
        if istart[0] != 0 | istart.size != 1:
            print("warning: more NaNs")
        ds = ds.isel(time=range(istop[0] + 1, ds.time.size))

    return ds


def cleanup_ud(ds):
    """More cleaning and calculation of derived variables."""

    # remove spikes in temperature
    for v in ["t1", "t2"]:
        ds[v] = despike(ds[v], ds.attrs["spike_thresh_t"])

    # despike T, C
    ibefore = 1
    iafter = 1
    for v in ["c1", "c2", "t1", "t2"]:
        ds[v].data = helpers.glitchcorrect(
            ds[v], ds.attrs[f"diff_{v[0]}"], ds.attrs[f"prod_{v[0]}"], ibefore, iafter
        )

    # Calculate salinity (both absolute and practical)
    ds = calcs.calc_sal(ds)

    # despike s
    ibefore = 2
    iafter = 2
    for v in ["s1", "s2", "SA1", "SA2"]:
        # remove spikes
        ds[v] = despike(ds[v], ds.attrs["spike_thresh_s"])
        # remove out of bounds data
        ds[v] = remove_out_of_bounds(
            ds[v], bmin=ds.attrs["bounds_s"][0], bmax=ds.attrs["bounds_s"][1]
        )
        ds[v].data = helpers.glitchcorrect(
            ds[v], ds.attrs["diff_s"], ds.attrs["prod_s"], ibefore, iafter
        )

    # calculate potential/conservative temperature, potential density anomaly
    ds = calcs.calc_temp(ds)
    ds = calcs.calc_sigma(ds)

    return ds


def despike(da, spike_threshold):
    """Set spikes to NaN."""
    absdiff = np.absolute(np.diff(da.data))
    # Using np.greater instead of the > operator as we can use the where option
    # and avoid the warning when nans are compared to a number. It broadcasts
    # to the original array size.
    ib = np.squeeze(
        np.where(np.greater(absdiff, spike_threshold, where=np.isfinite(absdiff)))
    )
    da[ib] = np.nan

    return da


def remove_out_of_bounds(da, bmin, bmax):
    """Remove out of bounds data."""
    ib = np.squeeze(
        np.where(
            (
                (np.greater(da, bmax, where=np.isfinite(da)))
                | (np.less(da, bmin, where=np.isfinite(da)))
            )
        )
    )
    da[ib] = np.nan

    return da


def preen_ctd(ds):
    """Remove spikes in p, t1, t2, c1, c2.

    Parameters
    ----------
    data : xarray.Dataset
            Dataset with CTD time series.

    Returns
    -------
    data : xarray.Dataset
            Cleaned dataset.
    """
    for v in ["p", "t1", "t2", "c1", "c2"]:
        ds[v] = ds[v].where(
            xr.zeros_like(ds[v]),
            helpers.preen(
                ds[v].data, ds.attrs[f"bounds_{v[0]}"][0], ds.attrs[f"bounds_{v[0]}"][1]
            ),
        )

    # TODO: remove spikes in oxygen, trans, fl in volts.
    return ds


def phase_correct(ds):
    """
    Bring temperature and conductivity in phase.

    Parameters
    ----------
    ds : dtype
            description

    Returns
    -------
    ds : dtype
            description
    """

    # remove spikes
    # TODO: bring this back in. however, the function fails later on if there
    # are nan's present. Could interpolate over anything that is just a few data points
    # for field in ["t1", "t2", "c1", "c2"]:
    # 	  ib = np.squeeze(np.where(np.absolute(np.diff(data[field].data)) > 0.5))
    # 	  data[field][ib] = np.nan

    # ---Spectral Analysis of Raw Data---
    # 24Hz data
    dt = 1 / 24
    # number of points per segment
    N = 2 ** 9

    # only data within tcfit range.
    ii = np.squeeze(
        np.argwhere(
            (ds.p.data > ds.attrs["tcfit"][0]) & (ds.p.data < ds.attrs["tcfit"][1])
        )
    )
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")
    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # fft of each segment (row). Data are detrended, then windowed.
    window = signal.triang(N) * np.ones((m, N))
    At1 = fft.fft(
        signal.detrend(np.reshape(ds.t1.data[i1:i2], newshape=(m, N))) * window
    )
    At2 = fft.fft(
        signal.detrend(np.reshape(ds.t2.data[i1:i2], newshape=(m, N))) * window
    )
    Ac1 = fft.fft(
        signal.detrend(np.reshape(ds.c1.data[i1:i2], newshape=(m, N))) * window
    )
    Ac2 = fft.fft(
        signal.detrend(np.reshape(ds.c2.data[i1:i2], newshape=(m, N))) * window
    )

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    At2 = At2[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]
    Ac2 = Ac2[:, 0 : int(N / 2)]

    # Frequency
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    f = f[: int(N / 2)]
    fold = f

    # Spectral Estimates. Note: In Matlab, At1*conj(At1) is not complex anymore.
    # Here, it is still a complex number but the imaginary part is zero.
    # We keep only the real part to stay consistent.
    Et1 = 2 * np.real(np.nanmean(At1 * np.conj(At1) / df / N ** 2, axis=0))
    Et2 = 2 * np.real(np.nanmean(At2 * np.conj(At2) / df / N ** 2, axis=0))
    Ec1 = 2 * np.real(np.nanmean(Ac1 * np.conj(Ac1) / df / N ** 2, axis=0))
    Ec2 = 2 * np.real(np.nanmean(Ac2 * np.conj(Ac2) / df / N ** 2, axis=0))

    # Cross Spectral Estimates
    Ct1c1 = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N ** 2, axis=0)
    Ct2c2 = 2 * np.nanmean(At2 * np.conj(Ac2) / df / N ** 2, axis=0)

    # Squared Coherence Estimates
    Coht1c1 = np.real(Ct1c1 * np.conj(Ct1c1) / (Et1 * Ec1))
    Coht2c2 = np.real(Ct2c2 * np.conj(Ct2c2) / (Et2 * Ec2))

    # Cross-spectral Phase Estimates
    Phit1c1 = np.arctan2(np.imag(Ct1c1), np.real(Ct1c1))
    Phit2c2 = np.arctan2(np.imag(Ct2c2), np.real(Ct2c2))

    # ---Determine tau and L---
    # tau is the thermistor time constant (sec)
    # L is the lag of t behind c due to sensor separation (sec)
    # Matrix of weights based on squared coherence.
    W1 = np.diag(Coht1c1)
    W2 = np.diag(Coht2c2)

    x1 = optimize.fmin(
        func=helpers.atanfit, x0=[0, 0], args=(f, Phit1c1, W1), disp=False
    )
    x2 = optimize.fmin(
        func=helpers.atanfit, x0=[0, 0], args=(f, Phit2c2, W2), disp=False
    )

    tau1 = x1[0]
    tau2 = x2[0]
    L1 = x1[1]
    L2 = x2[1]

    print("1: tau = {:1.4f}s, lag = {:1.4f}s".format(tau1, L1))
    print("2: tau = {:1.4f}s, lag = {:1.4f}s".format(tau2, L2))

    # ---Apply Phase Correction and LP Filter---
    ii = np.squeeze(np.argwhere(ds.p.data > 1))
    i1 = ii[0]
    i2 = ii[-1]
    n = i2 - i1 + 1
    n = (np.floor(n / N) * N).astype("int64")
    # Truncate to be multiple of N elements long
    i2 = (i1 + n).astype("int64")
    # number of segments = dof/2
    m = (n / N).astype("int64")

    # Number of degrees of freedom (power of 2)
    dof = 2 * m
    # Frequency resolution at dof degrees of freedom.
    df = 1 / (N * dt)

    # Transfer function
    f = fft.ifftshift(np.linspace(-N / 2, N / 2 - 1, N) / N / dt)
    H1 = (1 + 1j * 2 * np.pi * f * tau1) * np.exp(1j * 2 * np.pi * f * L1)
    H2 = (1 + 1j * 2 * np.pi * f * tau2) * np.exp(1j * 2 * np.pi * f * L2)

    # Low Pass Filter
    f0 = 6  # Cutoff frequency
    LP = 1 / (1 + (f / f0) ** 6)

    # Restructure data with overlapping segments.
    # Staggered segments
    vars = [
        "t1",
        "t2",
        "c1",
        "c2",
        "p",
        "trans",
        "fl",
        "par",
        "alt",
        "oxygen1",
        "oxygen2",
        "ph",
    ]
    vard = {}
    for v in vars:
        if v in ds:
            vard[v] = np.zeros((2 * m - 1, N))
            vard[v][: 2 * m - 1 : 2, :] = np.reshape(ds[v].data[i1:i2], newshape=(m, N))
            vard[v][1::2, :] = np.reshape(
                ds[v].data[i1 + int(N / 2) : i2 - int(N / 2)],
                newshape=(m - 1, N),
            )

    time = ds.time[i1:i2]
    lon = ds.lon[i1:i2]
    lat = ds.lat[i1:i2]

    # FFTs of staggered segments (each row)
    Ad = {}
    for v in vars:
        if v in ds:
            Ad[v] = fft.fft(vard[v])

    # Corrected Fourier transforms of temperature.
    Ad["t1"] = Ad["t1"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))
    Ad["t2"] = Ad["t2"] * ((H2 * LP) * np.ones((2 * m - 1, 1)))

    # LP filter remaining variables
    vars2 = [
        "c1",
        "c2",
        "p",
        "trans",
        "fl",
        "par",
        "alt",
        "oxygen1",
        "oxygen2",
        "ph",
    ]
    for v in vars2:
        if v in ds:
            Ad[v] = Ad[v] * (LP * np.ones((2 * m - 1, 1)))

    # Inverse transforms of corrected temperature
    # and low passed other variables
    Adi = {}
    for v in vars:
        if v in ds:
            Adi[v] = np.real(fft.ifft(Ad[v]))
            Adi[v] = np.squeeze(
                np.reshape(Adi[v][:, int(N / 4) : (3 * int(N / 4))], newshape=(1, -1))
            )

    time = time[int(N / 4) : -int(N / 4)]
    lon = lon[int(N / 4) : -int(N / 4)]
    lat = lat[int(N / 4) : -int(N / 4)]

    # Generate output structure. Copy attributes over.
    out = xr.Dataset(coords={"time": time})
    out.attrs = ds.attrs
    out["lon"] = lon
    out["lat"] = lat
    for v in vars:
        if v in ds:
            out[v] = xr.DataArray(Adi[v], coords=(out.time,))
            out[v].attrs = ds[v].attrs
    out.assign_attrs(dict(tau1=tau1, tau2=tau2, L1=L1, L2=L2))

    # ---Recalculate and replot spectra, coherence and phase---
    t1 = Adi["t1"][int(N / 4) : -int(N / 4)]  # Now N elements shorter
    t2 = Adi["t2"][int(N / 4) : -int(N / 4)]
    c1 = Adi["c1"][int(N / 4) : -int(N / 4)]
    c2 = Adi["c2"][int(N / 4) : -int(N / 4)]
    # p = Adi["p"][int(N / 4) : -int(N / 4)]

    m = (i2 - N) / N  # number of segments = dof/2
    m = np.floor(m).astype("int64")
    dof = 2 * m  # Number of degrees of freedom (power of 2)
    df = 1 / (N * dt)  # Frequency resolution at dof degrees of freedom.

    window = signal.triang(N) * np.ones((m, N))
    At1 = fft.fft(signal.detrend(np.reshape(t1, newshape=(m, N))) * window)
    At2 = fft.fft(signal.detrend(np.reshape(t2, newshape=(m, N))) * window)
    Ac1 = fft.fft(signal.detrend(np.reshape(c1, newshape=(m, N))) * window)
    Ac2 = fft.fft(signal.detrend(np.reshape(c2, newshape=(m, N))) * window)

    # Positive frequencies only
    At1 = At1[:, 0 : int(N / 2)]
    At2 = At2[:, 0 : int(N / 2)]
    Ac1 = Ac1[:, 0 : int(N / 2)]
    Ac2 = Ac2[:, 0 : int(N / 2)]
    fn = f[0 : int(N / 2)]

    Et1n = 2 * np.nanmean(np.absolute(At1[:, : int(N / 2)]) ** 2, 0) / df / N ** 2
    Et2n = 2 * np.nanmean(np.absolute(At2[:, : int(N / 2)]) ** 2, 0) / df / N ** 2
    Ec1n = 2 * np.nanmean(np.absolute(Ac1[:, : int(N / 2)]) ** 2, 0) / df / N ** 2
    Ec2n = 2 * np.nanmean(np.absolute(Ac2[:, : int(N / 2)]) ** 2, 0) / df / N ** 2

    # Cross Spectral Estimates
    Ct1c1n = 2 * np.nanmean(At1 * np.conj(Ac1) / df / N ** 2, axis=0)
    Ct2c2n = 2 * np.nanmean(At2 * np.conj(Ac2) / df / N ** 2, axis=0)

    # Squared Coherence Estimates
    Coht1c1n = np.real(Ct1c1n * np.conj(Ct1c1n) / (Et1n * Ec1n))
    Coht2c2n = np.real(Ct2c2n * np.conj(Ct2c2n) / (Et2n * Ec2n))
    # 95% confidence bound
    # epsCoht1c1n = np.sqrt(2) * (1 - Coht1c1n) / np.sqrt(Coht1c1n) / np.sqrt(m)
    # epsCoht2c2n = np.sqrt(2) * (1 - Coht2c2n) / np.sqrt(Coht2c2n) / np.sqrt(m)
    # 95% significance level for coherence from Gille notes
    betan = 1 - 0.05 ** (1 / (m - 1))

    # Cross-spectral Phase Estimates
    Phit1c1n = np.arctan2(np.imag(Ct1c1n), np.real(Ct1c1n))
    Phit2c2n = np.arctan2(np.imag(Ct2c2n), np.real(Ct2c2n))
    # 95% error bound
    # epsPhit1c1n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht1c1n) / (dof * Coht1c1n))
    # )
    # epsPhit1c2n = np.arcsin(
    # 	  stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht2c2n) / (dof * Coht2c2n))
    # )

    if ds.attrs["plot_spectra"]:
        fig, ax = plt.subplots(
            nrows=2, ncols=2, figsize=(9, 7), constrained_layout=True
        )
        ax0, ax1, ax2, ax3 = ax.flatten()

        ax0.plot(fold, Et1, label="1 uncorrected", color="0.5")
        ax0.plot(fold, Et2, label="2 uncorrected", color="0.8")
        ax0.plot(fn, Et1n, label="sensor 1")
        ax0.plot(fn, Et2n, label="sensor 2")
        ax0.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [$^{\circ}$C$^2$/Hz]",
            title="temperature spectra",
        )
        ax0.plot(
            [fn[50], fn[50]],
            [
                dof * Et1n[100] / stats.distributions.chi2.ppf(0.05 / 2, dof),
                dof * Et1n[100] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
            ],
            "k",
        )
        ax0.legend()

        ax1.plot(fold, Ec1, label="1 uncorrected", color="0.5")
        ax1.plot(fold, Ec2, label="2 uncorrected", color="0.8")
        ax1.plot(fn, Ec1n, label="1")
        ax1.plot(fn, Ec2n, label="2")
        ax1.set(
            yscale="log",
            xscale="log",
            xlabel="frequency [Hz]",
            ylabel=r"spectral density [mmho$^2$/cm$^2$/Hz]",
            title="conductivity spectra",
        )
        ax1.plot(
            [fn[50], fn[50]],
            [
                dof * Ec1n[100] / stats.distributions.chi2.ppf(0.05 / 2, dof),
                dof * Ec1n[100] / stats.distributions.chi2.ppf(1 - 0.05 / 2, dof),
            ],
            "k",
        )

        # Coherence between Temperature and Conductivity
        ax2.plot(fold, Coht1c1, color="0.5")
        ax2.plot(fold, Coht2c2, color="0.8")
        ax2.plot(fn, Coht1c1n)
        ax2.plot(fn, Coht2c2n)

        # ax.plot(fn, Coht1c1 / (1 + 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        # ax.plot(fn, Coht1c1 / (1 - 2 * epsCoht1c1), color="b", linewidth=0.5, alpha=0.2)
        ax2.plot(fn, betan * np.ones(fn.size), "k--")
        ax2.set(
            xlabel="frequency [Hz]",
            ylabel="squared coherence",
            ylim=(-0.1, 1.1),
            title="t/c coherence",
        )

        # Phase between Temperature and Conductivity
        ax3.plot(fold, Phit1c1, color="0.5")
        ax3.plot(fold, Phit2c2, color="0.8")
        ax3.plot(fn, Phit1c1n)
        ax3.plot(fn, Phit2c2n)
        ax3.set(
            xlabel="frequency [Hz]",
            ylabel="phase [rad]",
            ylim=[-4, 4],
            title="t/c phase",
            # 	  xscale="log",
        )
        ax3.plot(
            fold,
            -np.arctan(2 * np.pi * fold * x1[0]) - 2 * np.pi * fold * x1[1],
            "k--",
        )
        ax3.plot(
            fold,
            -np.arctan(2 * np.pi * fold * x2[0]) - 2 * np.pi * fold * x2[1],
            "k--",
        )

        if ds.attrs["plot_path"] != "":
            plt.savefig(ds.attrs["plot_path"], dpi=200)

    return out


def rmloops(ds):
    """
    Eliminate depth loops in CTD data based on sinking velocity.

    All data variables are set to NaN within depth loops.

    Parameters
    ----------
    ds : xarray.Dataset
            CTD time series dataset

    Returns
    -------
    ds : xarray.Dataset
            CTD time series dataset
    """

    assert "wthresh" in ds.attrs

    tsmooth = 0.25  # seconds
    fs = 24  # Hz
    w = calcs.wsink(ds.p.data, tsmooth, fs)  # down/up +/-ve
    iloop = np.array([])

    # up or downcast?
    pol = np.polyfit(np.array(range(ds.p.size)), ds.p, 1)
    dn = 1 if pol[0] > 0 else 0

    if dn:
        # downcast
        flp = np.squeeze(np.argwhere(w < ds.attrs["wthresh"]))
        if flp.size >= 1:
            ia, ib, ilen = helpers.findsegments(flp)
            if ia[0] == ib[0] == 0:
                ia = np.delete(ia, 0)
                ib = np.delete(ib, 0)
            for start, stop in zip(ia, ib):
                pmi = np.argmax(ds.p[:stop].data)
                pm = ds.p[pmi].data
                tmp = np.squeeze(np.where(ds.p[pmi:] < pm))
                iloop = np.append(iloop, pmi + 1)
                iloop = np.append(iloop, pmi + tmp - 1)
                iloop = np.unique(iloop)
            iloop = iloop.astype("int64")
        else:
            iloop = np.array([])
    else:
        # upcast
        flp = np.squeeze(np.argwhere(w > -ds.attrs["wthresh"]))
        if flp.size >= 1:
            ia, ib, ilen = helpers.findsegments(flp)
            if ia[0] == ib[0] == 0:
                ia = np.delete(ia, 0)
                ib = np.delete(ib, 0)
            for start, stop in zip(ia, ib):
                pmi = np.argmax(ds.p[:stop].data)
                pm = ds.p[pmi].data
                tmp = np.squeeze(np.where(ds.p[:pmi] < pm))
                iloop = np.append(iloop, pmi)
                iloop = np.append(iloop, pmi - tmp)
                iloop = np.unique(iloop)
            iloop = iloop.astype("int64")
        else:
            iloop = np.array([])

    # get data variable names
    varnames = [k for k, v in ds.data_vars.items()]
    for rmitem in ["lon", "lat", "z", "depth"]:
        if rmitem in varnames:
            varnames.remove(rmitem)
    # set loops to NaN
    for varn in varnames:
        ds[varn][iloop] = np.nan

    return ds


def bincast(ds, dz, zmin, zmax):
    """
    Depth-bin CTD time series.

    Parameters
    ----------
    ds : xr.Dataset
            CTD time series
    dz : float
            Bin size [m]
    zmin : float
            Minimum bin depth center [m]
    zmax : float
            Maximum bin depth center [m]

    Returns
    -------
    ds : xr.Dataset
            Depth-binned CTD profile
    """
    dz2 = dz / 2
    zbin = np.arange(zmin - dz2, zmax + dz + dz2, dz)
    zbinlabel = np.arange(zmin, zmax + dz, dz)

    # prepare dataset
    tmp = ds.swap_dims({"time": "depth"})
    tmp = tmp.reset_coords()

    # need to bin time separately, not sure why
    btime = tmp.time.groupby_bins(
        "depth", bins=zbin, labels=zbinlabel, right=True, include_lowest=True
    ).mean()
    # bin all variables
    out = tmp.groupby_bins(
        "depth", bins=zbin, labels=zbinlabel, right=True, include_lowest=True
    ).mean()

    # organize
    out.coords["time"] = btime
    out = out.set_coords(["lon", "lat"])
    out = out.rename({"depth_bins": "depth"})

    # copy attributes
    # get data variable names
    varnames = [k for k, v in ds.data_vars.items()]
    for vari in varnames:
        out[vari].attrs = ds[vari].attrs
    out["depth"].attrs = {"long_name": "depth", "units": "m"}
    out.attrs = ds.attrs

    # recalculate pressure from depth bins
    out["p"] = (("depth",), gsw.p_from_z(-1 * out.depth.data, out.lat.data))
    out.p.attrs = {"long_name": "pressure", "units": "dbar"}

    return out


def split_updn(ds):
    """
    Separate into down/up-casts and apply corrections.

    ## TODO: (should this really be here?)
    - tc lag correction
    - thermal mass correction
    - lowpass-filter T, C, oxygen

    Parameters
    ----------
    data : xarray.Dataset
            CTD time series

    Returns
    -------
    datad : xarray.Dataset
            CTD time series for downcast
    datau : xarray.Dataset
            CTD time series for upcast
    """
    n = ds.p.size
    ipmax = np.argmax(ds.p.data)

    datad = ds.isel(time=range(0, ipmax))
    datau = ds.isel(time=range(ipmax, n))

    return {"down": datad, "up": datau}
