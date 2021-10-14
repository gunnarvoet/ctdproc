import pathlib

import xarray as xr

import ctdproc as ctd


def test_success():
    assert True


# We defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_read_hex(rootdir, tmpdir):
    hexfile = rootdir / "data/BLT_Test_001.hex"
    assert type(hexfile) == pathlib.PosixPath
    print(hexfile)
    assert hexfile.exists()
    c = ctd.io.CTDHex(hexfile)

    cx = c.to_xarray()

    # make sure we can write and read the data as netcdf
    p = pathlib.Path(tmpdir) / "testfile.nc"
    cx.to_netcdf(p)
    cx2 = xr.open_dataset(p)
    assert type(cx2) == xr.core.dataset.Dataset


def test_read_lajit(rootdir, tmpdir):
    """Read test data from La Jolla Canyon."""
    hexfile = rootdir / "data/lajit2-sr1614-001.hex"
    assert type(hexfile) == pathlib.PosixPath
    print(hexfile)
    assert hexfile.exists()
    c = ctd.io.CTDHex(hexfile)

    cx = c.to_xarray()
    assert type(cx) == xr.core.dataset.Dataset
