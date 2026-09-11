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


def _up_leg_deeper_than_100dbar(lajit):
    """A towyo-like piece cut at the turnaround: starts at its own pressure
    maximum and never rises above 1 dbar."""
    up = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["up"]
    deep = np.flatnonzero(up.p.values > 100)
    return up.isel(time=deep)


def test_cleanup_accepts_a_piece_that_starts_at_its_deepest(lajit):
    piece = _up_leg_deeper_than_100dbar(lajit)
    first_time = piece.time.values[0]
    out = ctd.proc.cleanup(piece.copy(deep=True))
    assert out.time.values[0] == first_time


def test_run_profile_on_an_up_leg_that_starts_at_its_deepest(lajit):
    # cleanup (tested directly above) keeps this piece's deepest sample and
    # first time. run_profile's own phase_correct step then legitimately
    # trims a fixed N/4 = 128-sample margin off both ends as part of its
    # windowed FFT reconstruction (proc.py ~570), unrelated to either bug
    # fixed here, so the raw first sample is not expected to survive into
    # this output -- only that run_profile completes and returns usable
    # data for a piece that starts at its own pressure maximum.
    piece = _up_leg_deeper_than_100dbar(lajit)
    out = ctd.proc.run_profile(piece)
    assert out.sizes["time"] > 0
    assert np.isfinite(out.CT1.values[0])
    # The piece's start is the real turnaround of this cast, where the
    # package genuinely reverses; rmloops correctly NaNs that single loop.
    # Assert it is the only gap in CT1: one contiguous run, computed from
    # the data rather than pinned to rmloops's own loop-detection internals
    # (unchanged by this fix), with everything before and after finite.
    nan_idx = np.flatnonzero(np.isnan(out.CT1.values))
    assert nan_idx.size > 0
    assert np.array_equal(nan_idx, np.arange(nan_idx[0], nan_idx[-1] + 1))
    assert np.isfinite(out.CT1.values[: nan_idx[0]]).all()
    assert np.isfinite(out.CT1.values[nan_idx[-1] + 1 :]).all()


@pytest.mark.parametrize("direction", ["down", "up"])
def test_rmloops_on_a_leg_without_loops(direction):
    n = 24 * 100  # 100 s at 24 Hz
    p = np.linspace(0, 100, n)
    if direction == "up":
        p = p[::-1]
    time = np.datetime64("2020-01-01") + np.arange(n) * np.timedelta64(1000 // 24, "ms")
    ds = xr.Dataset(
        {"p": ("time", p), "t1": ("time", np.linspace(10, 5, n))},
        coords={"time": time},
    )
    ctd.io.add_default_proc_params(ds)
    before = ds.copy(deep=True)
    out = ctd.proc.rmloops(ds)
    xr.testing.assert_identical(out, before)
