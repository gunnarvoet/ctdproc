=======
ctdproc
=======


.. image:: https://img.shields.io/pypi/v/ctdproc.svg
        :target: https://pypi.python.org/pypi/ctdproc

.. image:: https://readthedocs.org/projects/ctdproc/badge/?version=latest
        :target: https://ctdproc.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
        :target: https://docs.astral.sh/ruff/
        :alt: Ruff



CTD data processing in python. 

* Free software: MIT license
* Documentation: https://ctdproc.readthedocs.io.


Features
--------

* Convert CTD data collected with Seabird 9/11 systems in hex-format to human-readable formats and physical units.

* Process CTD time series data into depth-binned profiles.


Additional Information
----------------------
Information on the Seabird hex data format saved by SBE 11 Deck Units can be found in the `SBE manual <./misc/manual-11pV2_018.pdf>`_ on p. 65ff.


Credits
-------

This package borrows heavily from a toolbox written in MATLAB® with contributions from Jennifer MacKinnon, Shaun Johnston, Daniel Rudnick, Robert Todd and others.


Docs
----
`uv run make docs` will generate the docs. 
