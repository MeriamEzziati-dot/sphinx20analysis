import numpy as np
import pytest
from astropy.io import fits

from MER_analysis.psf import PSF, PSFError


def random_array(number_of_pixels_x, number_of_pixels_y):
    data = np.random.random((number_of_pixels_x, number_of_pixels_y))
    return data


@pytest.fixture(scope="function")
def header():
    """Returns a dummy image fits header.

    Returns
    -------
    object
        A dummy image fits header.

    """
    # Create an empty astropy header object
    header = fits.header.Header()
    # Fill the header with the wcs information
    header.set("NAXIS", 2)
    header.set("NAXIS1", 50)
    header.set("NAXIS2", 50)
    return header


@pytest.fixture()
def image(header):
    return random_array(header["NAXIS1"], header["NAXIS2"])


@pytest.fixture()
def hdu_list(header, image):
    hdu_list = fits.HDUList(fits.PrimaryHDU(header=header, data=image))
    return hdu_list


@pytest.fixture()
def psf_fits_file(hdu_list, tmpdir):
    file = "psf.fits"
    filename = str(tmpdir.join(file))
    hdu_list.writeto(filename)
    return filename


@pytest.fixture()
def psf(psf_fits_file):
    return PSF.from_fits(psf_fits_file)


class TestPSF:
    """Tests for the PSF class"""

    def test_instanciation(self, header, image):
        psf = PSF(name="test_name", header=header, data=image)
        assert psf._name == "test_name"
        assert psf._header == header
        np.testing.assert_array_equal(psf._data, image)

    def test_from_fits(self, hdu_list, tmpdir):
        file = "psf.fits"
        filename = str(tmpdir.join(file))
        hdu_list.writeto(filename)
        psf = PSF.from_fits(filename)
        assert psf._name == file
        assert psf._header == hdu_list[0].header
        np.testing.assert_array_equal(psf._data, hdu_list[0].data)

    def test_get_name(self, monkeypatch, psf):
        monkeypatch.setattr(psf, "_name", "my_name")
        assert psf.get_name() == "my_name"

    def test_get_data(self, monkeypatch, psf, image):
        monkeypatch.setattr(psf, "_data", image)
        np.testing.assert_array_equal(psf.get_data(), image)

    def test_get_header(self, monkeypatch, psf, header):
        monkeypatch.setattr(psf, "_header", header)
        assert psf.get_header() == header

    def test_get_pixel_size(self, monkeypatch, psf):
        pixel_size = 0.1

        def mock_get_header():
            return {"CD1_1": pixel_size}

        monkeypatch.setattr(psf, "get_header", mock_get_header)
        assert psf.get_pixel_size() == 360
        assert psf.get_pixel_size(unit="degrees") == 0.1
        with pytest.raises(PSFError) as e:
            psf.get_pixel_size(unit="some_unit")
            assert e.message == "Unit must be either arcsec (default) or degrees"

    def test_normalize(self, psf):
        data = psf.get_data()
        psf.normalize()
        np.testing.assert_array_almost_equal(psf.get_data(), data / data.sum())

    def test_reduce_psf_bigger(self, psf):
        data = psf.get_data()
        array_dim = data.shape
        with pytest.raises(PSFError) as e:
            psf.reduce_psf((array_dim[0] + 1, array_dim[1] + 1))
            assert e.message == "The new dimensions should be smaller than the current array dimensions."

    def test_reduce_psf_same_size(self, psf):
        data = psf.get_data()
        array_dim = data.shape
        psf.reduce_psf(array_dim)
        np.testing.assert_array_equal(psf.get_data(), data)

    def test_reduce_psf_smaller(self, psf):
        data = psf.get_data()
        array_dim = data.shape
        psf.reduce_psf((array_dim[0] - 10, array_dim[1] - 10))
        np.testing.assert_array_equal(psf.get_data(), data[5:-5, 5:-5])

    def test_plot(self, psf, tmpdir):
        psf.plot(output_dir=tmpdir)
        p = tmpdir / psf.get_name().replace(".fits", ".png")
        assert p.exists()
