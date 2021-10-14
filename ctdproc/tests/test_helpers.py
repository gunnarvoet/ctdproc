import numpy as np
import pytest

from ctdproc import helpers


@pytest.fixture
def testdata():
    """Define a little test data set with random data from a
    Gaussion distribution. Inject spikes in two places.
    """
    testdata = np.random.randn(20)
    testdata[3:5] = 10
    testdata[12:15] = 20
    return testdata


@pytest.fixture
def testdata2():
    """Define a little test data set with random data from a
    Gaussion distribution. Inject spikes in two places.
    """
    x = np.random.randn(20) / 10
    x[5] = 50
    x[15] = 60
    return x


def test_unique_array():
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([1, 2, 3, 5])
    arr3 = np.array([1, 2])
    unique1 = helpers.unique_arrays(arr1, arr2)
    unique2 = helpers.unique_arrays(arr1, arr2, arr3)
    expect_out = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(unique1, expect_out)
    assert np.array_equal(unique2, expect_out)


def test_inearby(testdata):
    n = testdata.size
    ibad = np.squeeze(np.argwhere(np.absolute(testdata) > 5))
    assert np.array_equal(ibad, np.array([3, 4, 12, 13, 14]))
    assert np.array_equal(ibad, helpers.inearby(ibad, 0, 0, n))
    inrby1 = helpers.inearby(ibad, 1, 1, n)
    expect_out = np.array([2, 3, 4, 5, 11, 12, 13, 14, 15])
    assert np.array_equal(inrby1, expect_out)
    inrby2 = helpers.inearby(ibad, 10, 10, n)
    assert np.array_equal(testdata[inrby2], testdata)
    # also test for an empty ibad - should return and empty array
    assert helpers.inearby(np.array([]), 1, 1, n).size == 0


def test_findsegments(testdata):
    """Find segments in a given array of indices."""
    ibad = np.squeeze(np.argwhere(np.absolute(testdata) > 5))
    istart, istop, seglength = helpers.findsegments(ibad)
    assert seglength.size == 2
    assert np.array_equal(istart, np.array([3, 12]))
    assert np.array_equal(istop, np.array([4, 14]))


def test_interpbadsegments(testdata):
    """Interpolate over spikes in testdata."""
    ibad = np.squeeze(np.argwhere(np.absolute(testdata) > 5))
    y = helpers.interpbadsegments(testdata, ibad)
    assert np.all(y <= 5)


def test_glitchcorrect(testdata2):
    diffx = 2
    prodx = 1
    y = helpers.glitchcorrect(testdata2, diffx, prodx)
    assert np.all(np.absolute(y) < 10)


def test_preen(testdata):
    xp = helpers.preen(testdata, xmin=-5, xmax=5)
    assert np.all(np.absolute(xp) < 10)
