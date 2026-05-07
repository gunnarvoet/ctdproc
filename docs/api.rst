=============
API reference
=============

The public API is the four submodules re-exported from
:mod:`ctdproc`. The sections below group entries by their role in the
processing pipeline (:func:`ctdproc.proc.run_all`).

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
