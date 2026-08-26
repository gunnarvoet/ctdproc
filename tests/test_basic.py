import pathlib

import pytest
import xarray as xr
import numpy as np
import ctdproc as ctd
from munch import munchify


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


def test_pressure_temp_average_is_backward_looking():
    """The running mean must not see into the future."""
    pst = np.zeros(100)
    pst[50:] = 1.0
    avg = ctd.io._average_pressure_temp(pst, window_seconds=1.0, sample_rate=10.0)
    assert avg[49] == 0.0
    assert avg[50] == pytest.approx(0.1)
    assert avg[58] == pytest.approx(0.9)
    assert avg[59] == pytest.approx(1.0)


def test_pressure_temp_average_expands_at_start_of_record():
    """Fewer than one window of samples averages over what is available."""
    pst = np.arange(10, dtype="float64")
    avg = ctd.io._average_pressure_temp(pst, window_seconds=1.0, sample_rate=10.0)
    assert not np.any(np.isnan(avg))
    assert avg[0] == pytest.approx(0.0)
    assert avg[3] == pytest.approx(1.5)


def test_pressure_uses_averaged_pressure_temp(rootdir):
    """Pressure is computed from averaged, not instantaneous, pressure temperature."""
    hexfile = rootdir / "data/BLT_Test_001.hex"
    c = ctd.io.CTDHex(hexfile)
    p_instantaneous = (
        c._freq2pressure(c.dataraw.p, c.dataraw.pst, c.cfgp.PressureSensor.cal)
        - c._p_atm
    )
    assert not np.allclose(c.data.p, p_instantaneous)
    assert np.max(np.abs(c.data.p - p_instantaneous)) < 0.1


def test_altimeter_conversion_matches_seabird(rootdir):
    """Altimeter height is 300 * volts / scale factor + offset."""
    hexfile = rootdir / "data/BLT_Test_001.hex"
    c = ctd.io.CTDHex(hexfile)
    acal = c.cfgp.AltimeterSensor.cal
    # all test fixtures use the standard 15.000 scale factor
    assert acal.ScaleFactor == 15.0
    volt = np.array([0.0, 2.5, 5.0])
    alt = c._volt2alt(volt, acal)
    # a 0-5 V altimeter at the standard scale factor is a 100 m unit
    np.testing.assert_allclose(alt, [0.0, 50.0, 100.0])


def test_altimeter_conversion_applies_offset(rootdir):
    """The offset is added after scaling."""
    hexfile = rootdir / "data/BLT_Test_001.hex"
    c = ctd.io.CTDHex(hexfile)
    acal = munchify({"ScaleFactor": 30.0, "Offset": 2.0})
    np.testing.assert_allclose(c._volt2alt(np.array([3.0]), acal), [32.0])
