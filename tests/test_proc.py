import pathlib
import warnings

import gsw
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


def test_a_piece_too_short_for_the_phase_fit_raises_piece_too_short(lajit):
    down = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["down"]
    deep = np.flatnonzero(down.p.values > 300)[:400]
    assert issubclass(ctd.proc.PieceTooShort, ValueError)
    with pytest.raises(ctd.proc.PieceTooShort, match="too short for the phase fit"):
        ctd.proc.run_profile(down.isel(time=deep), clean=False)


@pytest.mark.parametrize("si", ["1", "2"])
def test_conservative_temperature_is_computed_from_absolute_salinity(lajit, si):
    down = ctd.proc.run_all(lajit.copy(deep=True))["down"]
    SA, SP, t, p = (down[v].values for v in (f"SA{si}", f"s{si}", f"t{si}", "p"))
    from_sa = gsw.CT_from_t(SA, t, p)
    from_sp = gsw.CT_from_t(SP, t, p)
    # SA - SP is 0.15 to 0.18 g/kg on this file, so CT from SP is at least
    # 0.002 degC warmer and the two are told apart
    assert np.nanmin(np.abs(from_sp - from_sa)) > 1e-4
    np.testing.assert_array_equal(down[f"CT{si}"].values, from_sa)


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


def _synthetic_deep_piece(n=24 * 200, nan_c=None):
    """A clean synthetic down-cast that never rises above 1 dbar, so
    cleanup's near-surface trim is a no-op and sample indices survive it.
    ``nan_c`` is a slice of c1 and c2 set to NaN; preen_ctd leaves NaN
    alone, so it reaches the NaN-trim block at the end of cleanup."""
    time = np.datetime64("2020-01-01") + np.arange(n) * np.timedelta64(1000 // 24, "ms")
    ds = xr.Dataset(
        {
            "p": ("time", np.linspace(5.0, 200.0, n)),
            "t1": ("time", np.linspace(20.0, 5.0, n)),
            "t2": ("time", np.linspace(20.0, 5.0, n)),
            "c1": ("time", np.linspace(5.0, 3.3, n)),
            "c2": ("time", np.linspace(5.0, 3.3, n)),
        },
        coords={"time": time},
    )
    ctd.io.add_default_proc_params(ds)
    if nan_c is not None:
        for v in ("c1", "c2"):
            ds[v][nan_c] = np.nan
    return ds


def test_cleanup_drops_a_leading_run_of_nan_conductivity():
    ds = _synthetic_deep_piece(nan_c=slice(0, 50))
    before = ds.copy(deep=True)
    out = ctd.proc.cleanup(ds)
    assert out.time.values[0] == before.time.values[50]
    assert out.sizes["time"] == before.sizes["time"] - 50


@pytest.mark.parametrize("start,stop", [(1000, 1050), (24 * 200 - 50, 24 * 200)])
def test_cleanup_keeps_nan_conductivity_that_is_not_a_leading_run(start, stop, capsys):
    """An interior or trailing run of NaN conductivity leaves the piece
    whole. Trimming to the end of the first run would drop every sample
    before it, and for a trailing run that is the whole piece."""
    ds = _synthetic_deep_piece(nan_c=slice(start, stop))
    before = ds.copy(deep=True)
    out = ctd.proc.cleanup(ds)
    assert out.sizes["time"] == before.sizes["time"]
    assert out.time.values[0] == before.time.values[0]
    assert "more NaNs" in capsys.readouterr().out


def test_cleanup_survives_a_single_nan_conductivity_sample():
    ds = _synthetic_deep_piece(nan_c=slice(1000, 1001))
    out = ctd.proc.cleanup(ds)
    assert out.sizes["time"] == 24 * 200


def test_run_all_names_a_channel_with_no_sample_in_bounds():
    """A deck test has conductivity near zero, outside ``bounds_c``."""
    ds = ctd.io.CTDx(LAJIT.parent / "ar73_dt001.hex")
    with pytest.raises(ValueError, match="no c1 sample within bounds_c"):
        ctd.proc.run_all(ds)


# run_all output on BLT_Test_001.hex. A change here is a change in processed
# values and needs a HISTORY entry.
BLT_REFERENCE = {
    "down": {
        "n": 58624,
        "nan": {"t1": 10413, "c1": 10413, "s1": 10399, "s2": 10397, "oxygen1": 10413},
        "mean": {
            "p": 575.742599,
            "t1": 7.107426,
            "s1": 34.251215,
            "s2": 34.255514,
            "CT1": 7.062551,
            "SA1": 34.422868,
            "sg1": 26.779401,
            "oxygen1": 1.014206,
        },
    },
    "up": {
        "n": 60672,
        "nan": {"t1": 66, "c1": 66, "s1": 66, "s2": 66, "oxygen1": 66},
        "mean": {
            "p": 579.871483,
            "t1": 7.089075,
            "s1": 34.245890,
            "s2": 34.250272,
            "CT1": 7.044358,
            "SA1": 34.417521,
            "sg1": 26.775393,
            "oxygen1": 1.032663,
        },
    },
}


@pytest.fixture(scope="module")
def blt_updown():
    return ctd.proc.run_all(ctd.io.CTDx(LAJIT.parent / "BLT_Test_001.hex"))


@pytest.mark.parametrize("direction", ["down", "up"])
def test_run_all_reproduces_the_blt_reference(blt_updown, direction):
    ds = blt_updown[direction]
    ref = BLT_REFERENCE[direction]
    assert ds.sizes["time"] == ref["n"]
    nan = {v: int(np.isnan(ds[v]).sum()) for v in ref["nan"]}
    assert nan == ref["nan"]
    mean = {v: float(np.nanmean(ds[v])) for v in ref["mean"]}
    assert mean == pytest.approx(ref["mean"], abs=2e-6)


@pytest.mark.parametrize("direction", ["down", "up"])
def test_salinity_cleanup_adds_no_nan_on_blt(blt_updown, direction):
    """Glitch correction of salinity once spread a despiked NaN over the
    segment next to it, 343 samples of s1 on this downcast."""
    ds = blt_updown[direction]
    for n in "12":
        assert np.isnan(ds[f"s{n}"]).sum() <= np.isnan(ds[f"t{n}"]).sum()


def test_run_profile_with_shallow_samples_ahead_of_the_piece(lajit):
    """phase_correct starts at the first sample below 1 dbar. Its segment
    count for the recomputed spectra must not depend on where that is."""
    down = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["down"]
    down = down.copy(deep=True)
    down.p[:600] = 0.5
    out = ctd.proc.run_profile(down, clean=False)
    assert out.sizes["time"] > 0


def test_conservative_temperature_where_absolute_salinity_exceeds_bounds_s():
    """Practical salinity 37.9 is inside ``bounds_s`` of 20 to 38 while its
    absolute salinity is above 38. CT must not go NaN there."""
    n = 24 * 20
    t, p = np.full(n, 13.0), np.linspace(900.0, 1000.0, n)
    c = gsw.C_from_SP(np.full(n, 37.9), t, p) / 10
    time = np.datetime64("2020-01-01") + np.arange(n) * np.timedelta64(1000 // 24, "ms")
    ds = xr.Dataset(
        {
            v: ("time", x.copy())
            for v, x in (("p", p), ("t1", t), ("t2", t), ("c1", c), ("c2", c))
        },
        coords={"time": time},
    )
    ds["lon"] = ("time", np.full(n, -10.0))
    ds["lat"] = ("time", np.full(n, 36.0))
    ctd.io.add_default_proc_params(ds)
    out = ctd.proc.cleanup_ud(ds)
    assert float(out.SA1.min()) > 38
    for v in ("s1", "SA1", "CT1", "sg1", "s2", "SA2", "CT2", "sg2"):
        assert np.isfinite(out[v]).all(), v


def _shallow_piece(lajit):
    """The part of the down-cast above 45 dbar, shallower than the default
    lower bound of the t/c fit range."""
    down = ctd.proc.split_updn(ctd.proc.cleanup(lajit.copy(deep=True)))["down"]
    return down.isel(time=np.flatnonzero(down.p.values < 45))


def test_run_profile_takes_the_fit_range_from_its_tcfit_argument(lajit):
    piece = _shallow_piece(lajit)
    # a fit range left on the piece by its parent cast must not be used
    piece.attrs["tcfit"] = [500, 1000]
    out = ctd.proc.run_profile(piece, clean=False, tcfit=[5, 45])
    assert out.attrs["tcfit"] == [5, 45]
    assert piece.attrs["tcfit"] == [500, 1000]
    assert np.isfinite(out.attrs["tau1"])


def test_piece_too_short_for_the_default_fit_range_names_tcfit(lajit):
    with pytest.raises(ctd.proc.PieceTooShort, match="tcfit"):
        ctd.proc.run_profile(_shallow_piece(lajit), clean=False)
