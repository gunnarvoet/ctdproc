=====
Usage
=====

The following is a minimal usage example. A more detailed example is located in the notebooks directory as a jupyter notebook. Start by importing ctdproc (and any other package you might need)::

    import numpy as np
    import ctdproc

Converting hex data
-------------------

Locate your raw CTD data in hex format, for example::

    hexfile = 'path/to/data.hex

As a first processing step, convert the raw hex-file into ascii data. Note that the corresponding xmlcon-file needs to be located in the same directory as the hex-file::

    c = ctdproc.io.CTD(hexfile)

.. note::

   Still need to add an option to include sensor configuration as an optional parameter input file as for some datasets there is no xml file on hand.

This generates a class instance that holds the time series from each sensor and the sensor configuration. Access the sensor config via::

    c.cfgp

Save the raw time series as Matlab file::

    c.to_mat('dataraw.mat')

Quickly convert the data into an xarray_ Dataset_. From this, save to netcdf format::

    cx = c.to_xarray()
    cx.to_netcdf('dataraw.nc')

.. _xarray: http://xarray.pydata.org/en/stable/
.. _Dataset: http://xarray.pydata.org/en/stable/data-structures.html#dataset

Data cleaning & sensor alignment
--------------------------------

Run further processing steps on the time series, including despiking and and corrections for thermal mass of the conductivity cell and misalignment between temperature and conductivity sensors. Data are also split into down- and up-cast. The following function conveniently combines a number of the processing steps::

    datad, datau = ctdproc.proc.run_all(cx)

Binning
-------

Finally, depth-bin the data::

    dz = 1
    zmin = 10
    zmax = np.ceil(datad.depth.max().data)
    datad = ctdproc.proc.ctd_bincast(datad, dz, zmin, zmax)
    datau = ctdproc.proc.ctd_bincast(datau, dz, zmin, zmax)
