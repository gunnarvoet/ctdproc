import datetime
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def unique_arrays(*arrays):
    """
    Find unique elements in more than one numpy array.

    Parameters
    ----------
    arrays : A tuple of np.array.ndarray

    Returns
    -------
    unique : np.array
        numpy array with unique elements from the input arrays
    """
    return np.unique(np.hstack(arrays))


def findsegments(ibad):
    """
    Find contiguous segments in an array of indices.

    Parameters
    ----------
    ibad : np.array
        Array with indices

    Returns
    -------
    istart : np.array
        Segment start indices
    istop : np.array
        Segment stop indices
    seglength : np.array
        Segment length
    """
    dibad = np.diff(ibad)
    jj = np.argwhere(dibad > 1)

    istart = jj + 1
    istart = np.insert(istart, 0, 0)
    istart = ibad[istart]

    istop = jj
    istop = np.hstack([np.squeeze(istop), ibad.size - 1])
    istop = ibad[istop]

    seglength = istop - istart + 1

    return istart, istop, seglength


def inearby(ibad, inearm, inearp, n):
    """
    Find indices left and right of given indices and add them to
    the index array.

    Differs from inearby.m as it breaks up ibad into segments and
    then finds nearby indices for each segment.

    Parameters
    ----------
    ibad : np.array
        Array with indices
    inearm : int
        Number of elements to include on left side
    inearp : int
        Number of elements to include on right side
    n : int
        Maximum index in output array and size of array
        being indexed

    Returns
    -------
    k : np.array
        New array of indices
    """
    if ibad.size == 0:
        k = np.array([]).astype("int64")
    else:
        istart, istop, seglength = findsegments(ibad)
        new_ind = np.array([]).astype("int64")
        for ia, ib in zip(istart, istop):
            new_ind = np.append(new_ind, np.arange(ia - inearm, ib + inearp + 1))
        k = np.unique(new_ind)
        k = k[((k >= 0) & (k < n))]
    return k


def interpbadsegments(x, ibad):
    """
    Interpolate over segments of bad data.

    Parameters
    ----------
    x : np.array
        Input data
    ibad : np.array
        Indices of bad data

    Returns
    -------
    y : np.array
        Interpolated array
    """

    def start_end_warning(loc):
        warnings.warn(
            message=f"no interpolation at {loc}",
            category=RuntimeWarning,
        )

    istart, istop, seglen = findsegments(ibad)
    y = x.copy()
    for iia, iis, iilen in zip(istart, istop, seglen):
        i1 = iia - 1
        i2 = range(iia, iis + 1)
        i3 = iis + 1
        if i1 < 0:
            start_end_warning("start")
        elif i3 > x.size:
            start_end_warning("end")
        else:
            y[i2] = interp1d(np.array([i1, i3]), x[[i1, i3]])(i2)
    return y


def glitchcorrect(x, diffx, prodx, ibefore=0, iafter=0):
    """
    Remove glitches/spikes in array.

    Adapted from tms_tc_glitchcorrect.m

    Parameters
    ----------
    x : np.array
        Input array
    diffx : int
        Threshold for differences
    prodx : int
        Threshold for products
    ibefore : int, optional
        Number of elements to interpolate on left side of glitch
    after : int, optional
        Number of elements to interpolate on right side of glitch

    Returns
    -------
    y : np.array
        Interpolated array
    """
    dx = np.diff(x)
    nx = len(dx)
    y = x.copy()

    with warnings.catch_warnings():
        # Prevent warning due to nans present in nanmin being printed
        warnings.simplefilter("ignore")
        dmin2 = np.nanmin(
            np.vstack([np.absolute(dx[0:-1]), np.absolute(dx[1:])]), axis=0
        )
        dmin3 = np.nanmin(
            np.vstack([np.absolute(dx[0:-2]), np.absolute(dx[2:])]), axis=0
        )

    dmul2 = -dx[0:-1] * dx[1:]
    dmul3 = -dx[0:-2] * dx[2:]

    ii2 = np.squeeze(
        np.argwhere(
            np.greater(dmul2, prodx, where=np.isfinite(dmul2))
            & np.greater(dmin2, diffx, where=np.isfinite(dmin2))
        )
    )
    ii3 = np.squeeze(
        np.argwhere(
            np.greater(dmul3, prodx, where=np.isfinite(dmul3))
            & np.greater(dmin3, diffx, where=np.isfinite(dmin3))
        )
    )

    ii2 = unique_arrays(ii2, ii2 + 1)
    ii3 = unique_arrays(ii3, ii3 + 1, ii3 + 2)

    jj2 = inearby(ii2, ibefore, iafter, nx)
    jj3 = inearby(ii3, ibefore, iafter, nx)

    jj = unique_arrays(jj2, jj3)

    if jj.size > 0:
        y = interpbadsegments(x, jj)

    return y


def preen(x, xmin, xmax):
    """
    Eliminate values outside given range and interpolate.

    Parameters
    ----------
    x : np.array
        Input array
    xmin : float
        Lower limit
    xmax : float
        Upper limit

    Returns
    -------
    xp : np.array
        Cleaned array
    """
    indexall = np.array(range(0, x.size))
    ii = np.squeeze(np.where(((x < xmin) | (x > xmax) | (np.imag(x) != 0))))
    indexclean = np.delete(indexall, ii)
    x = np.delete(x, ii)
    fint = interp1d(indexclean, x, bounds_error=False, fill_value=np.nan)
    xp = fint(indexall)
    return xp


def atanfit(x, f, Phi, W):
    f = np.arctan(2 * np.pi * f * x[0]) + 2 * np.pi * f * x[1] + Phi
    f = np.matmul(np.matmul(f.transpose(), W ** 4), f)
    return f


def pad_lr(p, nPad):
    """Pad array left and right"""
    p0 = p[0]
    p = p - p0
    p = p0 + np.insert(p, 0, -p[nPad - 1 :: -1])

    p0 = p[-1]
    p = p - p0
    p = p0 + np.insert(p, -1, -p[: -nPad - 1 : -1])

    return p


def mtlb2datetime(matlab_datenum, strip_microseconds=False, strip_seconds=False):
    """
    Convert Matlab datenum format to python datetime.
    This version also works for vector input and strips
    milliseconds if desired.

    Parameters
    ----------
    matlab_datenum : float or np.array
        Matlab time vector.
    strip_microseconds : bool
        Get rid of microseconds (optional)
    strip_seconds : bool
        Get rid of seconds (optional)

    Returns
    -------
    t : np.datetime64
        Time in numpy's datetime64 format.
    """

    if np.size(matlab_datenum) == 1:
        day = datetime.datetime.fromordinal(int(matlab_datenum))
        dayfrac = datetime.timedelta(days=matlab_datenum % 1) - datetime.timedelta(
            days=366
        )
        t1 = day + dayfrac
        if strip_microseconds and strip_seconds:
            t1 = datetime.datetime.replace(t1, microsecond=0, second=0)
        elif strip_microseconds:
            t1 = datetime.datetime.replace(t1, microsecond=0)

    else:
        t1 = np.ones_like(matlab_datenum) * np.nan
        t1 = t1.tolist()
        nonan = np.isfinite(matlab_datenum)
        md = matlab_datenum[nonan]
        day = [datetime.datetime.fromordinal(int(tval)) for tval in md]
        dayfrac = [
            datetime.timedelta(days=tval % 1) - datetime.timedelta(days=366)
            for tval in md
        ]
        tt = [day1 + dayfrac1 for day1, dayfrac1 in zip(day, dayfrac)]
        if strip_microseconds and strip_seconds:
            tt = [
                datetime.datetime.replace(tval, microsecond=0, second=0) for tval in tt
            ]
        elif strip_microseconds:
            tt = [datetime.datetime.replace(tval, microsecond=0) for tval in tt]
        tt = [np.datetime64(ti) for ti in tt]
        xi = np.where(nonan)[0]
        for i, ii in enumerate(xi):
            t1[ii] = tt[i]
        xi = np.where(~nonan)[0]
        for i in xi:
            t1[i] = np.datetime64("nat")
        t1 = np.array(t1)

    return t1


def datetime2mtlb(dt):
    pt = pd.to_datetime(dt)
    dt = pt.to_pydatetime()
    mdn = dt + datetime.timedelta(days=366)
    frac_seconds = [
        (dti - datetime.datetime(dti.year, dti.month, dti.day, 0, 0, 0)).seconds
        / (24.0 * 60.0 * 60.0)
        for dti in dt
    ]
    frac_microseconds = [
        dti.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0) for dti in dt
    ]
    out = np.array([mdni.toordinal() for mdni in mdn])
    out = out.astype(float) + frac_seconds + frac_microseconds
    return out
