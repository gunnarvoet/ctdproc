#!/usr/bin/env python
# coding: utf-8
import numpy as np
import xarray as xr
import gsw
from scipy import signal
from . import helpers


def swcalcs(data):
    """
    Calculate SA, s, CT, theta, sigma, oxygen, depth, z

    Using gsw package. Variable names for practical salinity
    and potential temperature are kept consistent with the
    Matlab processing package.

    Salinity is calculated both as absolute salinity (SA)
    and practical salinity (s). Note: Per TEOS-10 instructions,
    the variable to be archived is practical salinity.

    Temperature is converted to conservative temperature (CT)
    and potential temperature (th). Potential temperature
    is referenced to 0 dbar.

    Potential density anomaly is referenced to 0 dbar pressure.
    The anomaly is calculated as potential density minus 1000 kg/m^3.

    Parameters
    ----------
    data : array-like
        CTD time series data structure.

    Returns
    -------
    data : array-like
        CTD time series data structure.
    """
    # Most derived variables are now calculated in proc.ctd_cleanup2()
    # Not sure why they are calculated two times in the Matlab package.
    
    # data = calc_sal(data)
    # data = calc_temp(data)
    # data = calc_sigma(data)
    data = calc_depth(data)

    # TODO: oxygen

    return data


def calc_sal(data):
    # Salinity
    SA1, SP1 = calc_allsal(data.c1, data.t1, data.p, data.lon, data.lat)
    SA2, SP2 = calc_allsal(data.c2, data.t2, data.p, data.lon, data.lat)

    # Absolute salinity
    data["SA1"] = (["time"], SA1, {"long_name": "absolute salinity", "units": "g/kg"})
    data["SA2"] = (["time"], SA2, {"long_name": "absolute salinity", "units": "g/kg"})

    # Practical salinity
    data["s1"] = (["time"], SP1, {"long_name": "practical salinity", "units": ""})
    data["s2"] = (["time"], SP2, {"long_name": "practical salinity", "units": ""})

    return data


def calc_temp(data):
    # Conservative temperature
    for si in ["1", "2"]:
        data["CT{:s}".format(si)] = (
            ["time"],
            gsw.CT_from_t(data["s{:s}".format(si)], data["t{:s}".format(si)], data.p),
            {"long_name": "conservative temperature", "units": "°C"},
        )

    # Potential temperature
    for si in ["1", "2"]:
        data["th{:s}".format(si)] = (
            ["time"],
            gsw.pt_from_t(
                data["SA{:s}".format(si)], data["t{:s}".format(si)], p=data.p, p_ref=0
            ),
            {"long_name": "potential temperature", "units": "°C"},
        )

    return data


def calc_sigma(data):
    # Potential density anomaly
    for si in ["1", "2"]:
        data["sg{:s}".format(si)] = (
            ["time"],
            gsw.sigma0(data["SA{:s}".format(si)], data["CT{:s}".format(si)],),
            {"long_name": "potential density anomaly", "units": "kg/m$^3$"},
        )
    return data


def calc_depth(data):
    # Depth
    data.coords["depth"] = (
        ["time"],
        -1 * gsw.z_from_p(data.p, data.lat),
        {"long_name": "depth", "units": "m"},
    )
    return data


def calc_allsal(c, t, p, lon, lat):
    """
    Calculate absolute and practical salinity.

    Wrapper for gsw functions. Converts conductivity
    from S/m to mS/cm if output salinity is less than 5.

    Parameters
    ----------
    c : array-like
        Conductivity. See notes on units above.
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure, dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees
    Returns
    -------
    SA : array-like, g/kg
        Absolute Salinity
    SP : array-like
        Practical Salinity
    """
    SP = gsw.SP_from_C(c, t, p)
    if np.nanmean(SP) < 5:
        SP = gsw.SP_from_C(10 * c, t, p)
    SA = gsw.SA_from_SP(SP, p, lon, lat)
    return SA, SP


def wsink(p, Ts, Fs):
    """
    Compute sinking velocity from pressure record.

    Computes the sinking (or rising) velocity from the pressure signal p
    by first differencing. The pressure signal is smoothed with a low-pass
    filter for differentiation. If the input signal is shorter than the
    smoothing time scale, w is taken as the slope of the linear regression of p. 

    Adapted from wsink.m - Fabian Wolk, Rockland Oceanographic Services Inc.

    Parameters
    ----------
    p : array-like
        Pressure [dbar]
    Ts : float
        Smoothing time scale [s]
    Fs : float
        Sampling frequency [Hz]

    Returns
    -------
    w : array-like
        Sinking velocity [dbar/s]
    """
    FORDER = 1
    # low pass filter coefficients
    [b, a] = signal.butter(FORDER, 1 / Ts * 2 / Fs)
    N = p.size
    if N <= Fs * Ts * FORDER:
        pol = np.polyfit(np.array(range(N)), p, 1)
        w = pol[0] * Fs * np.ones(N)
    else:
        # pad the pressure vector left and right
        nPad = int(FORDER * Ts * Fs)
        if nPad > N:
            print(
                "warning: length of pressure vector is smaller than padding length.\n",
                "Filter transients may occur.",
            )
        p = helpers.pad_lr(p, nPad)
        w = np.gradient(Fs * signal.filtfilt(b, a, p))
        w = w[nPad:-nPad]

    return w
