from MER_analysis.maps import MERMosaic, MERMosaicError

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.wcs import WCS
import numpy as np
import pytest


@pytest.fixture()
def test_fits_image_filename():
    return "http://www.astropy.org/astropy-data/tutorials/FITS-images/HorseHead.fits"


@pytest.fixture()
def horsehead_coordinates():
    return SkyCoord('05 40 59.0 -02 27 30', unit=(u.hourangle, u.deg))


class TestEuclidMap:
    """Tests for the EuclidMap class"""

    def test_from_filename(self, test_fits_image_filename):
        filename = test_fits_image_filename
        euclid_map = MERMosaic.from_filename(filename)
        data = fits.getdata(filename)
        np.testing.assert_array_equal(euclid_map.data, data)
        header = fits.getheader(filename)
        wcs = WCS(header)
        assert euclid_map.wcs.__dict__ == wcs.__dict__
        assert euclid_map.band is None

    def test_extract_stamp_at_position(self, test_fits_image_filename, horsehead_coordinates, stamp_size=10):
        coordinates = horsehead_coordinates

        filename = test_fits_image_filename
        euclid_map = MERMosaic.from_filename(filename)
        ra, dec = coordinates.ra.deg, coordinates.dec.deg
        stamp = euclid_map.extract_stamp_at_position(ra, dec, stamp_size)

        data = fits.getdata(filename)
        header = fits.getheader(filename)
        wcs = WCS(header)
        position = wcs.world_to_pixel(coordinates)
        size = u.Quantity((stamp_size, stamp_size), u.pixel)
        astropy_stamp = Cutout2D(data, position, size)
        np.testing.assert_array_equal(stamp, astropy_stamp.data)

        ra, dec = 0., 0.
        with pytest.raises(MERMosaicError) as excinfo:
            stamp = euclid_map.extract_stamp_at_position(ra, dec, stamp_size)
            assert str(excinfo.value) == 'Source outside the map'



