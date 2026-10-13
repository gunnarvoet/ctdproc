=======
History
=======

.. 0. (unreleased)
.. ------------------

.. New Features
.. ~~~~~~~~~~~~

.. Bug Fixes
.. ~~~~~~~~~

.. Documentation
.. ~~~~~~~~~~~~~

.. Internal Changes
.. ~~~~~~~~~~~~~~~~


0.3.0 (2026-10-13)
------------------

This release changes processed values. Output of 0.2.0 differs from this release in the
following variables. Reprocess where they matter. Bug Fixes below has the details.

* ``CT1``, ``CT2``, ``sg1``, ``sg2``: conservative temperature was 2 to 7 mK too warm, more in
  warm water, and potential density anomaly 0.0002 to 0.0004 kg m-3 too light, in every
  release since 0.1.1.
* ``alt``: height above bottom was 3/4 of its value.
* ``oxygen1``: the sensor time response correction was 1/24 of its size.
* ``p``: changes by order 0.01 dbar, which can move the split between down- and upcast.
* Every variable: repeated runs of 0.2.0 on one file could give different output. Runs are
  now repeatable.

New Features
~~~~~~~~~~~~
* ``proc.run_profile`` processes one single-direction piece the caller has cut (half of a
  cast, a yo, one towyo leg) with the per-direction chain of ``run_all``, which now calls
  it. The refactor leaves ``run_all`` output unchanged. A piece under 1024 scans (43 s) in the t/c fit range
  raises ``PieceTooShort``. The ``tcfit`` argument sets the fit range, which by default starts
  at 50 dbar or deeper and so excludes a piece shallower than 50 dbar.
  (:commit:`3e0b530`, :commit:`e8e3dbc`)
* ``proc.PieceTooShort``, a subclass of ``ValueError``, marks a piece too short for the phase
  fit, so a caller can skip such pieces and let any other ``ValueError`` propagate.
  (:commit:`5f5041e`)

Breaking Changes
~~~~~~~~~~~~~~~~
* Removed unused plotting stubs ``ctdproc.proc.plot_profile`` and ``ctdproc.proc.plot_TS``. Use ``xarray.DataArray.plot()`` directly. (:issue:`9`)
* Moved ``wsink`` from ``ctdproc.calcs`` to ``ctdproc.proc`` — it is a signal-processing helper, not a TEOS-10 wrapper. Update imports from ``ctdproc.calcs.wsink`` to ``ctdproc.proc.wsink``. (:issue:`9`)
* ``ctdproc.helpers.atanfit`` and ``ctdproc.helpers.pad_lr`` are no longer part of the public API; they have been inlined as private helpers in ``ctdproc.proc`` where they are used. (:issue:`9`)

Bug Fixes
~~~~~~~~~
* The ``units`` attribute of ``c1`` and ``c2`` reads ``S m-1``. ``_freq2cond`` evaluates the
  Sea-Bird calibration equation with its factor 1/10, which yields S/m (about 3.6 in
  seawater), while the attribute and the ``physicalunits`` docstring said mS/cm.
  ``calcs.calc_allsal`` already converts to mS/cm before ``gsw.SP_from_C``, so salinity was
  right. Only the metadata changes. (:commit:`7708d93`)
* ``calcs.calc_temp`` computes conservative temperature from absolute salinity. It passed
  practical salinity to ``gsw.CT_from_t`` in every release from 0.1.1 to 0.2.0, so ``CT1``
  and ``CT2`` came out too warm, and the potential density anomalies ``sg1`` and ``sg2``,
  computed from CT, too light. On RR1708 cast 003 (yosonar level 5, fine grid) the error is
  0.0020 degC median in CT (5th to 95th percentile 0.0019 to 0.0023) and -0.00018 kg m-3
  median in sigma0. On ``lajit2-sr1614-001.hex`` it is 0.0025 to 0.0026 degC and -0.00034
  to -0.00036 kg m-3. The error grows with temperature. For practical salinity 34.7,
  ``gsw.CT_from_t`` gives 2.3 mK at 1.5 degC, 4.1 mK at 15 degC and 6.7 mK at 28 degC.
  Potential temperature ``th1``, ``th2`` and absolute salinity were
  right. Output of ``run_all``, ``run_profile`` and ``bincast`` changes in ``CT1``, ``CT2``,
  ``sg1`` and ``sg2``. (:commit:`2427f06`)
* Convert SBE43 oxygen after the time vector exists. ``CTDHex.physicalunits`` converted oxygen first and handed ``_volt2oxygen`` the scan index as time in seconds, so the tau*dV/dt term came out 1/24 of its size, from a 5-scan derivative window. On RR1708 ``FLEAT_recovery_002.hex`` the corrected ``oxygen1`` differs by 0.008 ml/L median absolute, 0.066 at the 95th and 0.14 at the 99th percentile, against a median of 2.27 ml/L. (:commit:`40ac350`)
* Initialize the output of the NaN-masked comparisons in ``helpers.glitchcorrect`` and ``proc.despike``. ``np.greater`` with ``where=`` and no ``out=`` leaves the masked entries uninitialized, so a comparison against NaN came back True at random and ``run_all`` returned different output on repeated runs of the same file: 1 to 11 samples per variable on ``lajit2-sr1614-001.hex``, up to 1.3e-4 in CT and sigma0, and moving NaN positions. On ``SKQ202417S_004.hex`` two runs of the code before this fix differed in 62 to 66 of the 1 m downcast bins, by up to 0.005 degC in ``t1`` and 0.008 degC in ``CT1``. (:commit:`a83b9e5`)
* ``helpers.interpbadsegments`` leaves a segment unchanged when the sample on either side of
  it is NaN. It interpolated between those two samples, so one NaN neighbor turned the whole
  segment into NaN. With the despiking fix above, ``glitchcorrect`` no longer counts a NaN as
  a glitch at random, and a sample that ``despike`` had set to NaN could border a flagged
  segment. On the upcast of ``SKQ202417S_004.hex``, 10 despiked ``s1`` samples in the
  thermocline grew to 393 NaN, which emptied 10 of the 1 m bins and left one bin 0.35 above
  its neighbors. The upcast now has 137 NaN ``s1`` samples where it had 520, and no emptied
  bin. Output of ``run_all`` on ``lajit2-sr1614-001.hex`` is unchanged. On
  ``BLT_Test_001.hex`` the downcast has 343 fewer NaN ``s1`` samples and 30 of its 1 m bins
  change, by up to 0.005 in ``s1``. (:commit:`e03aaff`)
* ``proc.preen_ctd`` raises a ``ValueError`` that names the channel and its bounds when no
  sample of ``p``, ``t1``, ``t2``, ``c1`` or ``c2`` is in range, as in a deck test or with a
  failed sensor. ``run_all`` on such a record failed inside ``scipy.interpolate.interp1d``
  with ``cannot reshape array of size 0``. The deck test ``ar73_dt001.hex``, with
  conductivity near 0.08 S/m against bounds of 2 to 6, now reports
  ``no c1 sample within bounds_c [2.0, 6.0]``. (:commit:`c215123`)
* Replace deprecated ``newshape`` keyword with ``shape`` in ``np.reshape`` calls in ``phase_correct`` for NumPy 2.x compatibility. (:pull:`55`)
* Compute pressure from a 30 second backward-looking running average of the pressure sensor temperature, as specified in the SBE Data Processing manual. Previously the instantaneous 12 bit pressure temperature word was used, so bit transitions caused small jumps in computed pressure. Computed pressure changes by order 0.01 dbar. The pressure maximum can land on a different scan, which moves the split between down- and upcast. On ``SKQ202417S_004.hex`` the upcast starts 0.54 s later. (:issue:`48`)
* Convert altimeter voltage with the Sea-Bird formula ``300 * volts / ScaleFactor + Offset``. The scale factor was previously applied as a multiplier, so with the standard scale factor of 15 the altimeter was read as a 75 m rather than a 100 m unit. Reported height above bottom changes by a factor of 4/3 for every reprocessed cast. Note that this corrects the conversion only; it does not address any remaining scale error in ``alt`` from the altimeter's internal sound speed assumption. (:issue:`60`)
* ``phase_correct`` keeps the fitted thermistor time constants and lags as ``tau1``,
  ``tau2``, ``L1``, ``L2`` in the output attrs. ``assign_attrs`` returned a new object that
  was discarded, and the output shared its attrs dict with the input. (:commit:`3e0b530`)
* ``proc.cleanup`` accepts a piece that starts at its deepest sample, such as a towyo up leg
  cut at the turnaround. The search for the start of the deep part excluded the index of the
  pressure maximum itself, so such a piece raised ``ValueError`` from ``np.max`` of an empty
  array. (:commit:`1f4becd`)
* ``proc.rmloops`` accepts a piece without loops. Its index of loop samples started as an
  empty float array and stayed one when no sample fell below ``wthresh``, which raised
  ``IndexError`` on assignment. It is now an empty integer array. (:commit:`1f4becd`)
* ``proc.cleanup`` trims a run of NaN conductivity only where the run starts at the first
  sample, as its docstring says. It trimmed to the end of the first run wherever that run
  sat, so a piece cut at depth lost every sample ahead of an interior run, and a piece whose
  conductivity goes out of range at the end came back empty. The guarding warning never fired
  in those cases: ``istart[0] != 0 | istart.size != 1`` binds as
  ``istart[0] != (0 | istart.size) != 1``, which is False for a single run anywhere but at
  index ``istart.size``. Reachable through ``run_profile`` on a towyo leg, and through
  ``preen_ctd``, which leaves NaN where it extrapolates past the first or last in-range
  sample. Output of ``run_all`` on ``lajit2-sr1614-001.hex`` is unchanged, verified variable
  by variable: that file has no NaN conductivity at the block, so the trim never ran. Nor
  does any of the 469 towyo leg pieces of RR1708 casts 002 to 004, checked one by one, so
  the yosonar level 5 written from them is unchanged as well. (:commit:`8bbd052`)
* ``proc.cleanup`` handles a single NaN conductivity sample. The index array came from
  ``np.squeeze(np.where(...))``, which collapses to 0-d for one hit, and
  ``helpers.findsegments`` raised ``ValueError`` from ``np.diff`` on it. It now uses
  ``np.flatnonzero``. (:commit:`8bbd052`)
* ``phase_correct`` counts the segments of its recomputed spectra from the length of the
  corrected series. It used the end index of the series, so a piece whose first sample below
  1 dbar sat at index 512 or later raised ``ValueError`` from ``np.reshape``. Reachable
  through ``run_profile`` with ``clean=False`` on a piece with 21 s or more of data above
  1 dbar ahead of it. Output is unchanged. (:commit:`d34ab07`)
* ``proc.cleanup_ud`` applies ``bounds_s`` to practical salinity, and ``SA1`` and ``SA2``
  take the bounds mask of their practical salinity. The bounds of 20 to 38 were applied to
  absolute salinity as well, which is about 0.18 g/kg higher, so ``SA`` was NaN at a
  practical salinity of 37.9. With the conservative temperature fix above, ``CT`` and ``sg``
  were NaN there too. (:commit:`092c64a`)
* ``helpers.interpbadsegments`` leaves a segment that ends on the last sample unchanged. It
  raised ``IndexError``. ``glitchcorrect`` never passes such a segment. (:commit:`10db7bd`)

Documentation
~~~~~~~~~~~~~
* Rebuild the API reference around a curated ``docs/api.rst`` grouped by pipeline stage, replacing the alphabetical autosummary dump generated by ``sphinx-apidoc``. Filled placeholder/missing docstrings on the public surface. (:issue:`9`)
* Fix the inconsistent heading level of "Building the documentation" in ``CONTRIBUTING.rst`` and drop the unused ``html_static_path`` entry from ``docs/conf.py`` so the Sphinx build is now warning-free. (:issue:`9`)

Internal Changes
~~~~~~~~~~~~~~~~
* Declare each submodule's public API explicitly via ``__all__`` (``proc``, ``io``, ``calcs``, ``helpers``). (:issue:`9`)
* Switch the Read the Docs build to uv-native install via ``method: uv`` and ``command: sync``, dropping the autogenerated ``docs/requirements.txt`` and the dependabot lane that kept it fresh. Split ``[dependency-groups]`` into ``dev`` and ``docs``. (:pull:`58`)


0.2.0 (2025-11-10)
------------------

New Features
~~~~~~~~~~~~
* Include oxygen sensor processing. (:pull:`49`)
  By `Alex Roberts <https://github.com/A-J-Roberts>`_.
* Move hard coded parameters to function inputs. (:pull:`20`)
* Include parameters and processing options as attributes in data structure. (Addressing :issue:`19`, :pull:`20`)
* Add CF standard names and units. (:pull:`25`)
  By `Jesse Cusack <https://github.com/jessecusack>`_.

Breaking Changes
~~~~~~~~~~~~~~~~
* Change the class name for the conversion from hex data. (:pull:`20`)
* Remove `z` dimension for binned casts. (:pull:`20`)
* Change allowed conductivity range from `[2.5, 6.0]` to `[2.0, 6.0]` to allow for Arctic surface waters to be processed. (:pull:`34`)

Bug Fixes
~~~~~~~~~
* Fix byte offset for reading hex file on R/V Armstrong (cruise AR73 May 2023). (:pull:`26`)
* Fix bug in `ctdproc.proc.remove_out_of_bounds()` for python > 3.8. (:pull:`29`)
  By `Øyvind Lundesgaard <https://github.com/oyvlun>`_.
* Fix hex reading bug introduced in PR26. (:pull:`43`)


.. Documentation
.. ~~~~~~~~~~~~~

Internal Changes
~~~~~~~~~~~~~~~~
* Rename some functions to be more self-explanatory. (:pull:`20`)
* Clean up variable names
* Remove unnecessary packages in conda development environment definition `environment.yml`. (:pull:`33`)


0.1.5 (2021-10-10)
------------------
Including all minor changes up to BLT2 cruise in October 2021.

New Features
~~~~~~~~~~~~
* Improve time handling.
* Add functions for saving profiles to .mat format.


0.1.4 (2020-04-20)
------------------

Bug Fixes
~~~~~~~~~
* Fix bug in pressure conversion, was missing subtraction of atmospheric pressure.
* Enable reading hex files with 48 bytes per scan. Fixes :issue:`11`.
* Enable reading hex files with 45 bytes per scan. Fixes :issue:`14`.

Documentation
~~~~~~~~~~~~~

* Add contributing guide. Mostly adapting `xarray's contributing guide <http://xarray.pydata.org/en/stable/contributing.html>`_ .


0.1.3 (2020-04-10)
------------------

* Fix more import issues

0.1.2 (2020-04-10)
------------------

* Fix import issues

0.1.1 (2020-04-10)
------------------

* First release on PyPI.
