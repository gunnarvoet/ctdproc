import pathlib

import xarray as xr
import numpy as np
import ctdproc as ctd


# We defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_read_hex(rootdir, tmpdir):
    hexfile = rootdir / "data/BLT_Test_001.hex"
    assert type(hexfile) is pathlib.PosixPath
    assert hexfile.exists()
    c = ctd.io.CTDHex(hexfile)
    cx = c.to_xarray()
    assert type(cx) is xr.core.dataset.Dataset

    # # make sure we can write and read the data as netcdf
    # # for some reason this results in weird warnings in pytest
    # # not doing this for now
    # p = pathlib.Path(tmpdir) / "testfile.nc"
    # cx.to_netcdf(p)
    # cx2 = xr.open_dataset(p)
    # assert type(cx2) == xr.core.dataset.Dataset


def test_read_lajit(rootdir):
    """Read test data from La Jolla Canyon."""
    hexfile = rootdir / "data/lajit2-sr1614-001.hex"
    assert type(hexfile) is pathlib.PosixPath
    assert hexfile.exists()
    c = ctd.io.CTDHex(hexfile)

    cx = c.to_xarray()
    assert type(cx) is xr.core.dataset.Dataset


def test_read_ar73(rootdir):
    """Read test data from cruise AR73."""
    hexfile = rootdir / "data/ar73_dt001.hex"
    assert type(hexfile) is pathlib.PosixPath
    print(hexfile)
    assert hexfile.exists()
    c = ctd.io.CTDHex(hexfile)
    _check_modcount_errors(c.data.modcount)

    cx = c.to_xarray()
    assert type(cx) is xr.core.dataset.Dataset


def _check_modcount_errors(modcount):
    """Check for modcount errors."""
    dmc = np.diff(modcount)
    mmc = np.mod(dmc, 256)
    fmc = np.squeeze(np.where(mmc - 1))
    assert len(fmc) == 0
