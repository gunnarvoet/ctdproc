#!/usr/bin/env python
# coding: utf-8
import numpy as np
import xarray as xr
from scipy import signal, optimize, fft, stats
import matplotlib as mpl

# We are running into trouble if the default backend is not installed
# in the current python environment. Try to import pyplot the default
# way first. If this fails, set backend to agg which should always work.
try:
    import matplotlib.pyplot as plt
except ImportError:
    mpl.use("agg")
    import matplotlib.pyplot as plt
import gsw
from . import helpers, calcs


def run_all(data):
    """
    Run all standard processing steps on raw CTD time series.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series data structure

    Returns
    -------
    datad : xarray.Dataset
        Downcast binned data
    datau : xarray.Dataset
        Upcast binned data
    """
    data = ctd_cleanup(data)

    datad, datau = ctd_correction_updn(data)

    tcfit = tcfit_default(data)
    datad = ctd_correction2(datad, tcfit)
    datau = ctd_correction2(datau, tcfit)

    datad = calcs.swcalcs(datad)
    datau = calcs.swcalcs(datau)

    wthresh = 0.1
    datad = ctd_rmloops(datad, wthresh)
    datau = ctd_rmloops(datau, wthresh)

    datad = ctd_cleanup2(datad)
    datau = ctd_cleanup2(datau)

    return datad, datau


def tcfit_default(data):
    """
    Get default values for tc fit range depending on depth of cast.

    Range for tc fit is 200dbar to maximum pressure if the cast is
    shallower than 1000dbar, 500dbar to max pressure otherwise.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series data structure

    Returns
    -------
    tcfit : tuple
        Upper and lower limit for tc fit in ctd_correction2.
    """
    if data.p.max() > 1000:
        tcfit = [500, data.p.max().data]
    else:
        tcfit = [200, data.p.max().data]
    return tcfit


def ctd_cleanup(data):
    """
    Clean up CTD raw time series.

    - despike pressure
    - eliminate data near surface
    - remove spikes in other data
    - remove smaller T, C, glitches

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series data structure

    Returns
    -------
    data : xarray.Dataset
        CTD time series data structure
    """

    # despike pressure
    diffp = 2
    prodp = 1
    data["p"].data = helpers.glitchcorrect(data.p.data, diffp, prodp)
    ipmax = np.argmax(data.p.data)

    # eliminate near-surface data
    ptop = 1
    fdeep = np.squeeze(np.where(data.p.data > ptop))
    ideepstart, ideepstop, ideeplen = helpers.findsegments(fdeep)
    ii = np.max(ideepstart[ideepstart < ipmax])
    jj = np.min(ideepstop[ideepstop >= ipmax])
    data = data.isel(time=range(ii, jj + 1))

    # remove spikes in temperature
    ib = np.squeeze(np.where(np.absolute(np.diff(data.t1.data)) > 0.5))
    data.t1[ib] = np.nan
    ib = np.squeeze(np.where(np.absolute(np.diff(data.t2.data)) > 0.5))
    data.t2[ib] = np.nan

    # remove out of range values
    data = ctd_preen(data)
    # no trans, fl ***

    # remove nans at start
    fnan1 = np.squeeze(np.where(np.isnan(data.c1.data)))
    fnan2 = np.squeeze(np.where(np.isnan(data.c2.data)))
    if np.array_equiv(fnan1, fnan2) is False:
        print("warning: NaNs index different in data.c1 and data.c2")
    if fnan1.size > 0:
        istart, istop, ilen = helpers.findsegments(fnan1)
        if istart[0] != 0 | istart.size != 1:
            print("warning: more NaNs")
        data = data.isel(time=range(istop[0] + 1, data.time.size))

    return data


def ctd_cleanup2(data):
    """More cleaning and calculation of derived variables."""

    # remove spikes in temperature
    data = ctd_despike(data, "t1", 0.5)
    data = ctd_despike(data, "t2", 0.5)

    # despike T, C
    prodc = 5e-7
    diffc = 1e-1
    prodt = 1e-4
    difft = 1e-1
    ibefore = 1
    iafter = 1
    data["c1"].data = helpers.glitchcorrect(data.c1, diffc, prodc, ibefore, iafter)
    data["c2"].data = helpers.glitchcorrect(data.c2, diffc, prodc, ibefore, iafter)
    data["t1"].data = helpers.glitchcorrect(data.t1, difft, prodt, ibefore, iafter)
    data["t2"].data = helpers.glitchcorrect(data.t2, difft, prodt, ibefore, iafter)

    # Calculate salinity (both absolute and practical)
    data = calcs.calc_sal(data)

    # despike s
    # remove spikes
    data = ctd_despike(data, "s1", 0.1)
    data = ctd_despike(data, "s2", 0.1)
    data = ctd_despike(data, "SA1", 0.1)
    data = ctd_despike(data, "SA2", 0.1)
    # remove out of bounds data
    data = ctd_remove_out_of_bounds(data, "s1", bmin=20, bmax=38)
    data = ctd_remove_out_of_bounds(data, "s2", bmin=20, bmax=38)
    data = ctd_remove_out_of_bounds(data, "SA1", bmin=20, bmax=38)
    data = ctd_remove_out_of_bounds(data, "SA2", bmin=20, bmax=38)
    # despike
    prods = 1e-8
    diffs = 1e-3
    ibefore = 2
    iafter = 2
    for vi in ["s1", "s2", "SA1", "SA2"]:
        data[vi].data = helpers.glitchcorrect(data[vi], diffs, prods, ibefore, iafter)

    # calculate potential/conservative temperature, potential density anomaly
    data = calcs.calc_temp(data)
    data = calcs.calc_sigma(data)

    return data


def ctd_despike(data, var, spike_threshold):
    """Set spikes to NaN."""
    absdiff = np.absolute(np.diff(data[var].data))
    # Using np.greater instead of the > operator as we can use the where option
    # and avoid the warning when nans are compared to a number. It broadcasts
    # to the original array size.
    ib = np.squeeze(
        np.where(np.greater(absdiff, spike_threshold, where=np.isfinite(absdiff)))
    )
    data[var][ib] = np.nan
    return data


def ctd_remove_out_of_bounds(data, var, bmin, bmax):
    """Remove out of bounds data."""
    ib = np.squeeze(
        np.where(
            (
                (np.greater(data[var], bmax, where=np.isfinite(data[var])))
                | (np.less(data[var], bmin, where=np.isfinite(data[var])))
            )
        )
    )
    data[var][ib] = np.nan
    return data


def ctd_preen(data):
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
    data["p"].data = helpers.preen(data.p.data, 0, 6200)
    data["t1"].data = helpers.preen(data.t1.data, -2, 40)
    data["t2"].data = helpers.preen(data.t2.data, -2, 40)
    data["c1"].data = helpers.preen(data.c1.data, 2.5, 6)
    data["c2"].data = helpers.preen(data.c2.data, 2.5, 6)
    # TODO: remove spikes in oxygen, trans, fl in volts.
    return data


def ctd_correction_updn(data):
    """
    Separate into down/up-casts and apply corrections.

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
    n = data.p.size
    ipmax = np.argmax(data.p.data)

    datad = data.isel(time=range(0, ipmax))
    datau = data.isel(time=range(ipmax, n))

    return datad, datau


def ctd_correction2(data, tcfit, plot_spectra=None, plot_path=None):
    """
    Bring temperature and conductivity in phase.

    Parameters
    ----------
    var : dtype
        description

    Returns
    -------
    var : dtype
        description
    """

    # remove spikes
    for field in ["t1", "t2", "c1", "c2"]:
        ib = np.squeeze(np.where(np.absolute(np.diff(data[field].data)) > 0.5))
        data[field][ib] = np.nan

    # ---Spectral Analysis of Raw Data---
    # 24Hz data
    dt = 1 / 24
    # number of points per segment
    N = 2 ** 9

    # only data within tcfit range.
    ii = np.squeeze(np.argwhere((data.p.data > tcfit[0]) & (data.p.data < tcfit[1])))
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
        signal.detrend(np.reshape(data.t1.data[i1:i2], newshape=(m, N))) * window
    )
    At2 = fft.fft(
        signal.detrend(np.reshape(data.t2.data[i1:i2], newshape=(m, N))) * window
    )
    Ac1 = fft.fft(
        signal.detrend(np.reshape(data.c1.data[i1:i2], newshape=(m, N))) * window
    )
    Ac2 = fft.fft(
        signal.detrend(np.reshape(data.c2.data[i1:i2], newshape=(m, N))) * window
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
    ii = np.squeeze(np.argwhere(data.p.data > 1))
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
    for vi in vars:
        if vi in data:
            vard[vi] = np.zeros((2 * m - 1, N))
            vard[vi][: 2 * m - 1 : 2, :] = np.reshape(
                data[vi].data[i1:i2], newshape=(m, N)
            )
            vard[vi][1::2, :] = np.reshape(
                data[vi].data[i1 + int(N / 2) : i2 - int(N / 2)], newshape=(m - 1, N)
            )

    time = data.time[i1:i2]
    lon = data.lon[i1:i2]
    lat = data.lat[i1:i2]

    # FFTs of staggered segments (each row)
    Ad = {}
    for vi in vars:
        if vi in data:
            Ad[vi] = fft.fft(vard[vi])

    # Corrected Fourier transforms of temperature.
    Ad["t1"] = Ad["t1"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))
    Ad["t2"] = Ad["t2"] * ((H1 * LP) * np.ones((2 * m - 1, 1)))

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
    for vi in vars2:
        if vi in data:
            Ad[vi] = Ad[vi] * (LP * np.ones((2 * m - 1, 1)))

    # Inverse transforms of corrected temperature
    # and low passed other variables
    Adi = {}
    for vi in vars:
        if vi in data:
            Adi[vi] = np.real(fft.ifft(Ad[vi]))
            Adi[vi] = np.squeeze(
                np.reshape(Adi[vi][:, int(N / 4) : (3 * int(N / 4))], newshape=(1, -1))
            )

    time = time[int(N / 4) : -int(N / 4)]
    lon = lon[int(N / 4) : -int(N / 4)]
    lat = lat[int(N / 4) : -int(N / 4)]

    # Generate output structure. Copy attributes over.
    out = xr.Dataset(coords={"time": time})
    out["lon"] = (["time"], lon)
    out["lat"] = (["time"], lat)
    for vi in vars:
        if vi in data:
            out[vi] = (["time"], Adi[vi])
            out[vi].attrs = data[vi].attrs
    out.attrs = dict(tau1=tau1, tau2=tau2, L1=L1, L2=L2)

    # ---Recalculate and replot spectra, coherence and phase---
    t1 = Adi["t1"][int(N / 4) : -int(N / 4)]  # Now N elements shorter
    t2 = Adi["t2"][int(N / 4) : -int(N / 4)]
    c1 = Adi["c1"][int(N / 4) : -int(N / 4)]
    c2 = Adi["c2"][int(N / 4) : -int(N / 4)]
    p = Adi["p"][int(N / 4) : -int(N / 4)]

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
    #     stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht1c1n) / (dof * Coht1c1n))
    # )
    # epsPhit1c2n = np.arcsin(
    #     stats.t.ppf(0.05, dof) * np.sqrt((1 - Coht2c2n) / (dof * Coht2c2n))
    # )

    if plot_spectra is not None:
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
            #     xscale="log",
        )
        ax3.plot(
            fold, -np.arctan(2 * np.pi * fold * x1[0]) - 2 * np.pi * fold * x1[1], "k--"
        )
        ax3.plot(
            fold, -np.arctan(2 * np.pi * fold * x2[0]) - 2 * np.pi * fold * x2[1], "k--"
        )

        if plot_path:
            plt.savefig(plot_path, dpi=200)

    return out


def ctd_rmloops(data, wthresh=0.1):
    """
    Eliminate depth loops in CTD data based on sinking velocity.

    All data variables are set to NaN within depth loops.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series dataset
    wthresh : float
        Sinking velocity threshold [m/s], default 0.1

    Returns
    -------
    data : xarray.Dataset
        CTD time series dataset
    """
    tsmooth = 0.25  # seconds
    fs = 24  # Hz
    pn = data.p.size
    w = calcs.wsink(data.p.data, tsmooth, fs)  # down/up +/-ve
    iloop = np.array([])

    # up or downcast?
    pol = np.polyfit(np.array(range(data.p.size)), data.p, 1)
    dn = 1 if pol[0] > 0 else 0

    if dn:
        # downcast
        flp = np.squeeze(np.argwhere(w < wthresh))
        if flp.size >= 1:
            ia, ib, ilen = helpers.findsegments(flp)
            nlp = ia.size
            for start, stop in zip(ia, ib):
                pmi = np.argmax(data.p[:stop]).data
                pm = data.p[pmi].data
                tmp = np.squeeze(np.where(data.p[pmi:] < pm))
                iloop = np.append(iloop, pmi + 1)
                iloop = np.append(iloop, pmi + tmp - 1)
            iloop = iloop.astype("int64")
        else:
            nlp = np.array([])
            iloop = np.array([])
    else:
        # upcast
        flp = np.squeeze(np.argwhere(w > -wthresh))
        if flp.size >= 1:
            ia, ib, ilen = helpers.findsegments(flp)
            nlp = ia.size
            for start, stop in zip(ia, ib):
                pmi = np.argmax(data.p[:stop]).data
                pm = data.p[pmi].data
                tmp = np.squeeze(np.where(data.p[:pmi] < pm))
                iloop = np.append(iloop, pmi)
                iloop = np.append(iloop, pmi - tmp)
            iloop = iloop.astype("int64")
        else:
            nlp = np.array([])
            iloop = np.array([])

    iloop2 = np.unique(iloop)

    # get data variable names
    varnames = [k for k, v in data.data_vars.items()]
    for rmitem in ["lon", "lat", "z", "depth"]:
        if rmitem in varnames:
            varnames.remove(rmitem)
    # set loops to NaN
    for varn in varnames:
        data[varn][iloop2] = np.nan

    return data


def ctd_bincast(data, dz, zmin, zmax):
    """
    Depth-bin CTD time series.

    Parameters
    ----------
    data : xr.Dataset
        CTD time series
    dz : float
        Bin size [m]
    zmin : float
        Minimum bin depth center [m]
    zmax : float
        Maximum bin depth center [m]

    Returns
    -------
    data : xr.Dataset
        Depth-binned CTD profile
    """
    dz2 = dz / 2
    zbin = np.arange(zmin - dz2, zmax + dz + dz2, dz)
    zbinlabel = np.arange(zmin, zmax + dz, dz)

    # prepare dataset
    tmp = data.swap_dims({"time": "depth"})
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
    out = out.rename_dims({"depth_bins": "z"})
    out = out.rename({"depth_bins": "depth"})

    # copy attributes
    # get data variable names
    varnames = [k for k, v in data.data_vars.items()]
    for vari in varnames:
        out[vari].attrs = data[vari].attrs
    out["depth"].attrs = {"long_name": "depth", "units": "m"}
    out.attrs = data.attrs

    # recalculate pressure from depth bins
    out["p"] = (["z"], gsw.p_from_z(-1 * out.depth, out.lat))
    out.p.attrs = {"long_name": "pressure", "units": "dbar"}

    return out
