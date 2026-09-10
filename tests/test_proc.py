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


def _pre_refactor_run_all(ds):
    """run_all as it stood in 0.2.0, the reference for the refactor."""
    ds = ctd.proc.cleanup(ds)
    ud = ctd.proc.split_updn(ds)
    for v, d in ud.items():
        ctd.proc.add_tcfit_default(d)
        d = d.dropna("time", how="any")
        d = ctd.proc.phase_correct(d)
        d = ctd.calcs.swcalcs(d)
        d = ctd.proc.rmloops(d)
        d = ctd.proc.cleanup_ud(d)
        ud[v] = d
    return ud


def test_run_all_matches_the_pre_refactor_chain(lajit):
    new = ctd.proc.run_all(lajit.copy(deep=True))
    old = _pre_refactor_run_all(lajit.copy(deep=True))
    for d in ("down", "up"):
        xr.testing.assert_equal(new[d], old[d])


def test_run_profile_equals_one_direction_of_run_all(lajit):
    down = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["down"]
    got = ctd.proc.run_profile(down, clean=False)
    xr.testing.assert_equal(got, ctd.proc.run_all(lajit.copy(deep=True))["down"])


def test_run_profile_does_not_modify_its_input(lajit):
    ds = lajit.copy(deep=True)
    # a slice isel shares memory and the attrs dict with ds
    piece = ds.isel(time=slice(0, ds.sizes["time"] // 2))
    before = ds.copy(deep=True)
    ctd.proc.run_profile(piece)
    xr.testing.assert_identical(ds, before)


def test_run_profile_rejects_a_piece_too_short_for_the_phase_fit(lajit):
    down = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["down"]
    deep = np.flatnonzero(down.p.values > 300)[:400]
    with pytest.raises(ValueError, match="too short for the phase fit"):
        ctd.proc.run_profile(down.isel(time=deep), clean=False)


def test_phase_correct_records_its_fit_per_direction(lajit):
    out = ctd.proc.run_all(lajit.copy(deep=True))
    for d in ("down", "up"):
        for k in ("tau1", "tau2", "L1", "L2"):
            assert np.isfinite(out[d].attrs[k])
    assert out["down"].attrs is not out["up"].attrs
    assert out["down"].attrs["tau1"] != out["up"].attrs["tau1"]
