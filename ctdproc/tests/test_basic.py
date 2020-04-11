import pathlib
import ctdproc as ctd

def test_success():
    assert True
    assert 1

# we defined rootdir as a fixture in conftest.py
# and can use it here as input now
def test_read_hex(rootdir, tmpdir):
    hexfile = (rootdir / 'data/BLT_Test_001.hex')
    assert type(hexfile)==pathlib.PosixPath
    print(hexfile)
    assert hexfile.exists()
    c = ctd.io.CTD(hexfile)

    cx = c.to_xarray()

    p = pathlib.Path(tmpdir) / 'testfile.nc'
    cx.to_netcdf(p)