# Improve API / docs autobuild (issue #9)

## Context

The Sphinx build at `docs/_build/html/` currently produces a single dumping-ground page (`source/ctdproc.html`, ~57KB, 700+ lines) that lists every submodule's contents alphabetically with `:undoc-members:` enabled. This means:

1. **Undocumented internals appear empty.** Functions like `plot_profile`, `plot_TS`, `atanfit`, `pad_lr`, and the `CTDHex.__init__` boilerplate show up as bare signatures with no body. Readers can't tell whether the function is private, deprecated, or just unfinished.
2. **There is no narrative.** The library has a clear pipeline (`run_all` → `cleanup` → `split_updn` → `phase_correct` → `swcalcs` → `rmloops` → `cleanup_ud` → `bincast`), but the API page lists everything alphabetically, so a new user can't map docstrings back to the workflow described in `usage.rst` and the project README.
3. **Several public docstrings are placeholders.** `proc.phase_correct` is annotated as `ds : dtype / description`; `proc.cleanup_ud`, `proc.despike`, and `proc.remove_out_of_bounds` are one-line summaries with no Parameters/Returns blocks; `io.CTDx`, `io.add_default_proc_params`, and most of `calcs.py` (`calc_sal`, `calc_temp`, `calc_sigma`, `calc_depth`) have no docstring at all.
4. **The public API is implicit.** No `__all__` anywhere — the only declaration of "what's public" is the re-export line in `__init__.py`, which lists modules, not symbols.

The intended outcome is an API reference that is genuinely useful when looking up a method or function: organized by purpose, free of empty stubs, with every listed symbol carrying a real Parameters/Returns block, and with the public surface declared explicitly in code.

## Approach

Four parallel workstreams:

1. **Targeted module-level refactor** (small, surgical) so the public surface is coherent before it gets documented.
2. **Define the public API** with `__all__` in each submodule.
3. **Replace the auto-generated API page** with a hand-curated `docs/api.rst` that uses `autosummary` + `autodoc`, grouped by pipeline stage.
4. **Fill missing/placeholder docstrings** on the public symbols only.

## 1. Targeted refactor

Confirmed call-site map (single message `grep` over `src/`):

- `calcs.wsink` is called from `proc.rmloops` only.
- `helpers.pad_lr` is called from `calcs.wsink` only.
- `helpers.atanfit` is called from `proc.phase_correct` only (two call sites).
- `proc.plot_profile` and `proc.plot_TS` have no callers in `src/` or `tests/`.

Changes:

- **Move `wsink`** from `src/ctdproc/calcs.py` → `src/ctdproc/proc.py`. It's a signal-processing function (low-pass-filtered first difference of pressure), not a TEOS-10 wrapper. Update `proc.rmloops` to call `wsink(...)` directly instead of `calcs.wsink(...)`. Drop the `from . import calcs` line from rmloops's reach if no longer needed; verify other call sites in proc.py still need calcs.
- **Inline `pad_lr` as `_pad_lr` in `proc.py`** alongside the moved `wsink`. Co-locate the private helper with its only consumer; remove from `helpers.py`. Drop the `from . import helpers` import in calcs.py if `pad_lr` was the only thing it used (it was).
- **Inline `atanfit` as `_atanfit` in `proc.py`** alongside `phase_correct`. Update the two `helpers.atanfit` references in `phase_correct` to bare `_atanfit`. Remove from `helpers.py`.
- **Delete `plot_profile` and `plot_TS`** from `proc.py`. Two-line stubs with no docstrings, no callers, and `xarray.DataArray.plot()` already gives users sensible defaults.

After this, `helpers.py` is purely a public utility module (segment-finding, glitch correction, datetime conversion) and `proc.py` owns its private signal-processing helpers. Module ownership is clear.

## 2. `__all__` declarations

Add to the top of each submodule. Internals (anything not listed, including `_atanfit` and `_pad_lr`) will not appear in the curated reference.

- `src/ctdproc/proc.py` →
  ```python
  __all__ = [
      "run_all", "cleanup", "cleanup_ud", "split_updn",
      "phase_correct", "rmloops", "bincast", "wsink",
      "despike", "remove_out_of_bounds", "preen_ctd",
      "add_tcfit_default",
  ]
  ```
- `src/ctdproc/io.py` → `["CTDHex", "CTDx", "add_default_proc_params", "prof_to_mat"]`
- `src/ctdproc/calcs.py` → `["swcalcs", "calc_sal", "calc_temp", "calc_sigma", "calc_depth", "calc_allsal"]` (no `wsink` — moved to proc)
- `src/ctdproc/helpers.py` → `["findsegments", "inearby", "interpbadsegments", "glitchcorrect", "preen", "unique_arrays", "mtlb2datetime", "datetime2mtlb"]` (no `pad_lr`, no `atanfit` — both moved to proc as private)

These match the symbols the curated docs will surface, so the package and the docs agree on the public surface.

## 2. Sphinx structure changes

### `docs/conf.py`

Add `sphinx.ext.autosummary` to `extensions` and enable stub generation:

```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",  # new — for xarray/numpy/gsw cross-refs
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # no "undoc-members" — undocumented symbols stay out
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "gsw": ("https://teos-10.github.io/GSW-Python/", None),
}
```

### Replace `docs/source/` with a curated `docs/api.rst`

Delete the autogeneration step and create a hand-written page grouped by pipeline stage. Skeleton:

```rst
=============
API reference
=============

Pipeline orchestration
----------------------
.. autosummary::
   :toctree: generated/

   ctdproc.proc.run_all
   ctdproc.proc.split_updn
   ctdproc.proc.bincast

Cleaning
--------
.. autosummary::
   :toctree: generated/

   ctdproc.proc.cleanup
   ctdproc.proc.cleanup_ud
   ctdproc.proc.despike
   ctdproc.proc.remove_out_of_bounds
   ctdproc.proc.preen_ctd
   ctdproc.proc.rmloops
   ctdproc.proc.wsink

Phase correction
----------------
.. autosummary::
   :toctree: generated/

   ctdproc.proc.phase_correct
   ctdproc.proc.add_tcfit_default

Thermodynamic calculations
--------------------------
.. autosummary::
   :toctree: generated/

   ctdproc.calcs.swcalcs
   ctdproc.calcs.calc_sal
   ctdproc.calcs.calc_temp
   ctdproc.calcs.calc_sigma
   ctdproc.calcs.calc_depth
   ctdproc.calcs.calc_allsal

I/O
---
.. autosummary::
   :toctree: generated/

   ctdproc.io.CTDHex
   ctdproc.io.CTDx
   ctdproc.io.add_default_proc_params
   ctdproc.io.prof_to_mat

Low-level utilities
-------------------
.. autosummary::
   :toctree: generated/

   ctdproc.helpers.findsegments
   ctdproc.helpers.inearby
   ctdproc.helpers.interpbadsegments
   ctdproc.helpers.glitchcorrect
   ctdproc.helpers.preen
   ctdproc.helpers.unique_arrays
   ctdproc.helpers.mtlb2datetime
   ctdproc.helpers.datetime2mtlb
```

`autosummary` with `:toctree: generated/` produces a one-line summary table on the API page and a separate detail page per symbol — much better navigation than today's giant single page.

### `docs/index.rst`

Replace `source/modules` with `api`:
```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing
   authors
   history
   notes
```

### `Makefile` `docs` target

Drop the `sphinx-apidoc` step and the `docs/source/` regeneration. Add `docs/generated/` to the clean step (autosummary stub output) and to `.gitignore`. New target:

```makefile
docs: ## generate Sphinx HTML documentation, including API docs
	rm -rf docs/source docs/generated docs/_build
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
```

Also remove `docs/source/` from the repo (currently has `ctdproc.rst` and `modules.rst` — both are stale autogenerated files).

## 3. Docstring pass on public API

NumPy style, per `CONTRIBUTING.rst`. Each docstring needs a one-line summary, an optional extended description, then `Parameters` / `Returns` (and `See Also` where useful for the pipeline). Specific functions to fill in or fix:

**`src/ctdproc/proc.py`**
- `phase_correct` (line 231) — replace `ds : dtype / description` template with real input/output description; add a brief explanatory paragraph (this is the FFT-based T-vs-C phase correction described in the Architecture section of `CLAUDE.md`).
- `cleanup_ud` (line 145) — expand from one line to full Parameters/Returns; mention that it runs after `swcalcs` and recomputes salinity-derived spikes.
- `despike` (line 184) — add Parameters/Returns; clarify NaN behavior.
- `remove_out_of_bounds` (line 198) — add Parameters/Returns.
- `add_tcfit_default` (line 67) — current docstring claims the function returns `tcfit`, but it actually mutates `ds.attrs["tcfit"]` and returns `None`. Fix.
- `run_all`, `cleanup`, `bincast`, `rmloops`, `split_updn`, `preen_ctd` — already have decent docstrings; minor copy-edits only (e.g., `run_all` documents `data` parameter but signature is `ds`).
- `wsink` (moved in from `calcs.py`) — keep the existing NumPy-style docstring; it's already in good shape.

**`src/ctdproc/io.py`**
- `CTDx` (line 19) — no docstring; add a short one explaining it is the convenience wrapper combining `CTDHex` + `to_xarray` + `add_default_proc_params`.
- `add_default_proc_params` (line 25) — no docstring; document that it mutates `ds.attrs` in place and list the parameters it sets (or at least their categories: bounds, spike thresholds, glitch-correction thresholds, sinking-velocity threshold, plotting flags).
- `CTDHex` class docstring (line 47) — currently has a TODO list but no usage example or Attributes block. Add a short example and document the public attributes (`filename`, `cfgp`, `dataraw`) and the four public methods.
- `CTDHex.parse_hex`, `read_xml_config`, `physicalunits`, `to_mat`, `to_xarray`, `sbetime_to_mattime`, `mattime_to_sbetime`, `mattime_to_datetime64` — add NumPy-style docstrings.
- `prof_to_mat` (line 986) — add docstring.

**`src/ctdproc/calcs.py`**
- `calc_sal`, `calc_temp`, `calc_sigma`, `calc_depth` (lines 52, 100, 135, 154) — currently no docstrings. Add a one-line summary plus Parameters/Returns. Mention which variables each adds to the dataset (`SA1/2`, `s1/2`, `CT1/2`, `th1/2`, `sg1/2`, `depth`).
- `swcalcs` — already has a good description; minor copy edits.

**`src/ctdproc/helpers.py`**
- `mtlb2datetime` and `datetime2mtlb` — `mtlb2datetime` has a decent docstring; `datetime2mtlb` has none — add a short Parameters/Returns block.
- `_atanfit`, `_pad_lr` — moved into `proc.py` as private; remain undocumented.

## Files to modify / create / delete

- **Modify**: `src/ctdproc/proc.py` — add `__all__`, delete `plot_profile`/`plot_TS`, add `wsink` (moved from calcs), add private `_atanfit`/`_pad_lr` (moved from helpers), update `phase_correct` and `rmloops` call sites, fill docstrings.
- **Modify**: `src/ctdproc/calcs.py` — add `__all__`, remove `wsink`, drop now-unused `from . import helpers` if applicable, fill docstrings on `calc_*`.
- **Modify**: `src/ctdproc/helpers.py` — add `__all__`, remove `atanfit` and `pad_lr`, expand `datetime2mtlb` docstring.
- **Modify**: `src/ctdproc/io.py` — add `__all__`, fill docstrings on `CTDx`, `add_default_proc_params`, `CTDHex` class and its public methods, `prof_to_mat`.
- **Modify**: `docs/conf.py` (extensions + autosummary settings).
- **Modify**: `docs/index.rst` (replace `source/modules` with `api`).
- **Modify**: `Makefile` (`docs` target).
- **Modify**: `.gitignore` (add `docs/generated/`).
- **Create**: `docs/api.rst`.
- **Delete**: `docs/source/ctdproc.rst`, `docs/source/modules.rst`, and the empty `docs/source/` directory.
- **Modify**: `HISTORY.rst` (add unreleased entry referencing `:issue:` 9 per project convention; mention removal of `plot_profile`/`plot_TS` and the `wsink` module move as user-visible changes).

## Reuse / existing utilities

- `napoleon` (numpy docstrings), `viewcode`, `extlinks` are already configured in `conf.py`; only add `autosummary` and `intersphinx`.
- `sphinx_rtd_theme` already in use; no theme change.
- `:issue:` and `:pull:` extlinks are already set up — use them in any new cross-references.

## Verification

1. `make docs` builds without warnings (especially no `WARNING: document isn't included in any toctree` from the deleted `source/` files; no `WARNING: autosummary: failed to import` for any listed symbol).
2. Open `docs/_build/html/index.html` and confirm:
   - The "API reference" entry appears in the sidebar.
   - The API page shows the six section headers (Pipeline orchestration, Cleaning, Phase correction, Thermodynamic calculations, I/O, Low-level utilities).
   - Each summary table row is a clickable link to a per-symbol detail page under `generated/`.
   - No symbol detail page is empty — every page shows a one-line summary, Parameters, and Returns.
   - `CTDHex` detail page lists the public methods (`parse_hex`, `read_xml_config`, `physicalunits`, `to_mat`, `to_xarray`) and not the underscore-prefixed internals.
3. `make format-check` and `make check` pass.
4. `make test` still passes. Existing tests touch only `ctdproc.io.CTDHex` and `ctdproc.helpers.{unique_arrays,inearby,findsegments,interpbadsegments,glitchcorrect,preen}` — none of the moved/deleted symbols, so no test edits expected.
5. Spot-check that intersphinx resolves: e.g., a `Parameters` block referencing `xarray.Dataset` should render as a link to `docs.xarray.dev`.
6. Sanity-check the refactor: `grep -rn "plot_profile\|plot_TS\|calcs.wsink\|helpers.atanfit\|helpers.pad_lr" src/ tests/` returns no hits.
