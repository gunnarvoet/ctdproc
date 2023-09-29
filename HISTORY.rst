=======
History
=======

0.1.7 (unreleased)
------------------

.. New Features
.. ~~~~~~~~~~~~

.. Bug Fixes
.. ~~~~~~~~~

.. Documentation
.. ~~~~~~~~~~~~~

.. Internal Changes
.. ~~~~~~~~~~~~~~~~



0.1.6 (unreleased)
------------------

New Features
~~~~~~~~~~~~
* Move hard coded parameters to function inputs. (:pull:`20`)
* Include parameters and processing options as attributes in data structure. (Addressing :issue:`19`, :pull:`20`)

Breaking Changes
~~~~~~~~~~~~~~~~
* Change the class name for the conversion from hex data. (:pull:`20`)
* Remove `z` dimension for binned casts. (:pull:`20`)
* Change allowed conductivity range from `[2.5, 6.0]` to `[2.0, 6.0]` to allow for Arctic surface waters to be processed. (:pull:`34`)

Bug Fixes
~~~~~~~~~
* Fix byte offset for reading hex file on R/V Armstrong (cruise AR73 May 2023). (:pull:`26`)

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

.. New Features
.. ~~~~~~~~~~~~

Bug Fixes
~~~~~~~~~
* Fix bug in pressure conversion, was missing subtraction of atmospheric pressure.
* Enable reading hex files with 48 bytes per scan. Fixes :issue:`11`.
* Enable reading hex files with 45 bytes per scan. Fixes :issue:`14`.

Documentation
~~~~~~~~~~~~~

* Add contributing guide. Mostly adapting `xarray's contributing guide <http://xarray.pydata.org/en/stable/contributing.html>`_ .

.. Internal Changes
.. ~~~~~~~~~~~~~~~~


0.1.3 (2020-04-10)
------------------

* Fix more import issues

0.1.2 (2020-04-10)
------------------

* Fix import issues

0.1.1 (2020-04-10)
------------------

* First release on PyPI.
