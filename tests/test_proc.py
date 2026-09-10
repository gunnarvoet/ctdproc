import pathlib
import warnings

import numpy as np
import pytest
import xarray as xr

import ctdproc as ctd

LAJIT = pathlib.Path(__file__).parent / "data" / "lajit2-sr1614-001.hex"


def test_glitchcorrect_with_nans_emits_no_uninitialized_memory_warning():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=2000)) * 0.01
    x[100:110] = np.nan
    x[500] += 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ctd.helpers.glitchcorrect(x, 0.1, 1e-4, 1, 1)


def test_despike_does_not_flag_samples_next_to_a_nan():
    x = np.r_[np.zeros(50), np.nan, np.zeros(50), 3.0, np.zeros(50)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = ctd.proc.despike(xr.DataArray(x.copy()), 0.5)
    # the jump into and out of the 3.0 spike flags samples 100 and 101
    assert np.isnan(out.values[100]) and np.isnan(out.values[101])
    # the differences touching the NaN at 50 are NaN and must not flag 49
    assert out.values[49] == 0.0


@pytest.fixture(scope="module")
def lajit():
    return ctd.io.CTDx(LAJIT)


def test_run_all_is_deterministic(lajit):
    a = ctd.proc.run_all(lajit.copy(deep=True))
    b = ctd.proc.run_all(lajit.copy(deep=True))
    for d in ("down", "up"):
        xr.testing.assert_equal(a[d], b[d])
