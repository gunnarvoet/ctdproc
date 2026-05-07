#!/usr/bin/env python
# coding: utf-8
import gsw
import numpy as np

__all__ = [
    "swcalcs",
    "calc_sal",
    "calc_temp",
    "calc_sigma",
    "calc_depth",
    "calc_allsal",
]


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
    """
    Add absolute and practical salinity for both sensor pairs.

    Adds variables ``SA1``, ``SA2`` (absolute salinity, g/kg) and ``s1``,
    ``s2`` (practical salinity) to the dataset, computed from
    conductivity, temperature, pressure, and the cast position via
    :func:`calc_allsal`.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series with ``c1``, ``c2``, ``t1``, ``t2``, ``p``,
        ``lon``, ``lat``.

    Returns
    -------
    xarray.Dataset
        Input augmented with ``SA1``, ``SA2``, ``s1``, ``s2``.
    """
    # Salinity
    SA1, SP1 = calc_allsal(data.c1, data.t1, data.p, data.lon, data.lat)
    SA2, SP2 = calc_allsal(data.c2, data.t2, data.p, data.lon, data.lat)

    # Absolute salinity
    data["SA1"] = (
        ("time",),
        SA1.data,
        {
            "long_name": "absolute salinity",
            "units": "g kg-1",
            "standard_name": "sea_water_absolute_salinity",
        },
    )
    data["SA2"] = (
        ("time",),
        SA2.data,
        {
            "long_name": "absolute salinity",
            "units": "g kg-1",
            "standard_name": "sea_water_absolute_salinity",
        },
    )

    # Practical salinity
    data["s1"] = (
        ("time",),
        SP1.data,
        {
            "long_name": "practical salinity",
            "units": "",
            "standard_name": "sea_water_practical_salinity",
        },
    )
    data["s2"] = (
        ("time",),
        SP2.data,
        {
            "long_name": "practical salinity",
            "units": "",
            "standard_name": "sea_water_practical_salinity",
        },
    )

    return data


def calc_temp(data):
    """
    Add conservative and potential temperature for both sensor pairs.

    Adds variables ``CT1``, ``CT2`` (conservative temperature, deg C) and
    ``th1``, ``th2`` (potential temperature, deg C, referenced to 0 dbar)
    to the dataset.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series with ``s1``/``s2``, ``SA1``/``SA2``, ``t1``/``t2``,
        and ``p``. Run :func:`calc_sal` first.

    Returns
    -------
    xarray.Dataset
        Input augmented with ``CT1``, ``CT2``, ``th1``, ``th2``.
    """
    # Conservative temperature
    for si in ["1", "2"]:
        data["CT{:s}".format(si)] = (
            ("time",),
            gsw.CT_from_t(
                data["s{:s}".format(si)], data["t{:s}".format(si)], data.p
            ).data,
            {
                "long_name": "conservative temperature",
                "units": "degree_C",
                "standard_name": "sea_water_conservative_temperature",
            },
        )

    # Potential temperature
    for si in ["1", "2"]:
        data["th{:s}".format(si)] = (
            ("time",),
            gsw.pt_from_t(
                data["SA{:s}".format(si)],
                data["t{:s}".format(si)],
                p=data.p,
                p_ref=0,
            ).data,
            {
                "long_name": "potential temperature",
                "units": "degree_C",
                "standard_name": "sea_water_potential_temperature",
            },
        )

    return data


def calc_sigma(data):
    """
    Add potential density anomaly for both sensor pairs.

    Adds variables ``sg1``, ``sg2`` (potential density anomaly, kg/m^3,
    referenced to 0 dbar) to the dataset.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series with ``SA1``/``SA2`` and ``CT1``/``CT2``. Run
        :func:`calc_sal` and :func:`calc_temp` first.

    Returns
    -------
    xarray.Dataset
        Input augmented with ``sg1``, ``sg2``.
    """
    # Potential density anomaly
    for si in ["1", "2"]:
        data["sg{:s}".format(si)] = (
            ("time",),
            gsw.sigma0(
                data["SA{:s}".format(si)],
                data["CT{:s}".format(si)],
            ).data,
            {
                "long_name": "potential density anomaly",
                "units": "kg m-3",
                "standard_name": "sea_water_sigma_theta",
                "reference_pressure": "0 dbar",
            },
        )
    return data


def calc_depth(data):
    """
    Add a ``depth`` coordinate (m, positive down) computed from pressure.

    Uses :func:`gsw.z_from_p` with latitude.

    Parameters
    ----------
    data : xarray.Dataset
        CTD time series with ``p`` and ``lat``.

    Returns
    -------
    xarray.Dataset
        Input with a new ``depth`` coordinate.
    """
    # Depth
    data.coords["depth"] = (
        ("time",),
        -1 * gsw.z_from_p(data.p, data.lat).data,
        {
            "long_name": "depth",
            "units": "m",
            "standard_name": "depth",
            "positive": "down",
        },
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
