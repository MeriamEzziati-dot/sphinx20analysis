from astropy.io import fits
from astropy.table import Table
import numpy as np
import pytest

from MER_analysis.catalogs import Catalog, MERCatalog
from MER_analysis.psf import VISGridPSF

@pytest.fixture(scope="function")
def dummy_header():
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
    header.set("NAXIS1", 1000)
    header.set("NAXIS2", 500)
    header.set("CTYPE1", "RA---TAN")
    header.set("CTYPE2", "DEC--TAN")
    header.set("CRVAL1", 233.738563)
    header.set("CRVAL2", 23.503139)
    header.set("CRPIX1", 500)
    header.set("CRPIX2", 250)
    header.set("CD1_1", -0.1 / 3600)
    header.set("CD1_2", 0.0)
    header.set("CD2_1", 0.0)
    header.set("CD2_2", 0.1 / 3600)

    return header


@pytest.fixture(scope="function")
def dummy_hdulist(dummy_header):
    """Returns a dummy object catalog fits.

    Parameters
    ----------
    image_header: object
        A dummy image fits header.

    Returns
    -------
    object
        A dummy MER object catalog fits.

    """
    # Create the HDU list
    hdu_list = fits.HDUList(fits.PrimaryHDU(header=dummy_header))

    # Create the catalog columns adding some random information
    n_sources = dummy_header['NAXIS2']
    radec_center = np.array([dummy_header["CRVAL1"], dummy_header["CRVAL2"]])
    magnitudes = np.random.random_sample(n_sources) * 15 - 30
    tu_fnu_vis = pow(10, - (magnitudes - 8.9) / 2.5)
    id_column = fits.Column(
        name="SOURCE_ID", array=np.arange(n_sources), format="K")
    ra_column = fits.Column(
        name="RA", array=radec_center[0] + 0.1 * np.random.random(
            n_sources), format="D")
    dec_column = fits.Column(
        name="DEC", array=radec_center[1] + 0.1 * np.random.random(
            n_sources), format="D")
    tu_fnu_vis_column = fits.Column(
        name='TU_FNU_VIS', array=tu_fnu_vis, format="D")

    catalog_table = fits.BinTableHDU.from_columns(
        [id_column, ra_column, dec_column, tu_fnu_vis_column], header=dummy_header)

    # Add the table to the HDU list
    hdu_list.append(catalog_table)

    return hdu_list


@pytest.fixture(scope="function")
def dummy_catalog_fits_file(dummy_hdulist, tmpdir):
    file = 'dummy_catalog.fits'
    filename = str(tmpdir.join(file))
    dummy_hdulist.writeto(filename)
    return filename


@pytest.fixture(scope="function")
def dummy_catalog(dummy_catalog_fits_file):
    return Catalog.from_file(dummy_catalog_fits_file)


@pytest.fixture(scope="function")
def dummy_astropy_table(dummy_hdulist):
    return Table(dummy_hdulist[1].data)


@pytest.fixture(scope="function")
def mer_hdulist():
    """Returns a dummy object catalog fits.

    Parameters
    ----------
    image_header: object
        A dummy image fits header.

    Returns
    -------
    object
        A dummy MER object catalog fits.

    """
    # Create the HDU list
    hdu_list = fits.HDUList(fits.PrimaryHDU())

    # Create the catalog columns adding some random information
    n_sources = 1000
    columns = []
    for column_name in MERCatalog.ORIGINAL_COLUMNS:
        columns.append(fits.Column(name=column_name, array=np.random.randn(n_sources), format='D'))
    columns.append(fits.Column(name='FLUX_DETECTION_TOTAL', array=np.random.randn(n_sources), format='D'))
    catalog_table = fits.BinTableHDU.from_columns(columns)

    # Add the table to the HDU list
    hdu_list.append(catalog_table)

    return hdu_list


@pytest.fixture(scope="function")
def mer_catalog_fits_file(mer_hdulist, tmpdir):
    file = 'mer_catalog.fits'
    filename = str(tmpdir.join(file))
    mer_hdulist.writeto(filename)
    return filename


@pytest.fixture(scope='function')
def vis_grid_psf_main_header():
    header = fits.header.Header()
    header['NEXTEND'] = 3
    return header


@pytest.fixture(scope='function')
def vis_grid_psf_image_header():
    header = fits.header.Header()
    header['EXTNAME'] = 'chip{}'.format(np.random.randint(9))
    header['NAXIS1'] = 180
    header['NAXIS2'] = 180
    return header


@pytest.fixture(scope='function')
def vis_grid_psf_image_data(vis_grid_psf_image_header):
    size_x = vis_grid_psf_image_header['NAXIS1']
    size_y = vis_grid_psf_image_header['NAXIS2']
    data = np.random.randn(size_x*size_y).reshape(size_x, size_y)
    return data


@pytest.fixture(scope='function')
def dummy_vis_grid_psf(vis_grid_psf_main_header, vis_grid_psf_image_header, vis_grid_psf_image_data):
    main_header = vis_grid_psf_main_header
    header = {vis_grid_psf_image_header['EXTNAME']: vis_grid_psf_image_header}
    data = {vis_grid_psf_image_header['EXTNAME']: vis_grid_psf_image_data}
    return VISGridPSF('dummy_vis_grid_psf.fits', main_header, header, data)