import pytest
from astropy.coordinates import SkyCoord
import numpy as np

from MER_analysis.catalogs import Catalog, CatalogTools


class TestCatalog:
    """A collection of tests for the Catalog class"""

    def test_from_file(self, tmpdir, dummy_hdulist):
        file = 'dummy_catalog.fits'
        filename = str(tmpdir.join(file))
        input_hdulist = dummy_hdulist
        input_hdulist.writeto(filename)
        catalog = Catalog.from_file(filename)
        assert input_hdulist[1].header == catalog.header
        np.testing.assert_equal(input_hdulist[1].data, catalog.data)

    def test_from_data(self, dummy_header, dummy_astropy_table):
        catalog = Catalog.from_data(dummy_header, dummy_astropy_table)
        assert dummy_header == catalog.header
        assert list(dummy_astropy_table.columns) == catalog.columns
        np.testing.assert_equal(dummy_astropy_table.as_array(), catalog.data.as_array())

    def test_extract_coords(self, dummy_catalog_fits_file):
        catalog = Catalog.from_file(dummy_catalog_fits_file)
        catalog.extract_coords()
        assert type(catalog.coords) == SkyCoord
        np.testing.assert_array_equal(catalog.data['RA'], catalog.coords.ra.data)
        np.testing.assert_array_equal(catalog.data['DEC'], catalog.coords.dec.data)

        catalog._data.remove_column('RA')
        with pytest.raises(NameError, match="The catalog has no RA and DEC columns"):
            catalog.extract_coords()

    def test_match(self, dummy_catalog_fits_file):
        catalog1 = Catalog.from_file(dummy_catalog_fits_file)
        catalog2 = Catalog.from_file(dummy_catalog_fits_file)
        catalog1.extract_coords()
        catalog2.extract_coords()
        idx, dist = catalog1.match(catalog2)
        np.testing.assert_array_equal(idx, np.arange(len(catalog1.data)))
        np.testing.assert_array_equal(dist, np.zeros(len(catalog1.data)))

    def test_append(self, dummy_catalog_fits_file):
        catalog1 = Catalog.from_file(dummy_catalog_fits_file)
        catalog2 = Catalog.from_file(dummy_catalog_fits_file)
        catalog1.extract_coords()
        catalog2.extract_coords()
        original_size = catalog1.header['NAXIS2']
        catalog1.append(catalog2)
        assert catalog1.header['NAXIS2'] == original_size * 2
        np.testing.assert_array_equal(
            catalog1.data[:original_size],
            catalog1.data[original_size:])
        np.testing.assert_array_equal(
            catalog1.coords.ra.data[:original_size],
            catalog1.coords.ra.data[original_size:])
        np.testing.assert_array_equal(
            catalog1.coords.dec.data[:original_size],
            catalog1.coords.dec.data[original_size:])


class TestCatalogTools:
    def test_create_new_header(self, dummy_catalog):
        cat1 = dummy_catalog
        cat2 = dummy_catalog
        header = CatalogTools.create_new_header(cat1, cat2)

        assert header['XTENSION'] == 'BINTABLE'
        assert header['BITPIX'] == 8
        assert header['NAXIS'] == 2
        assert header['NAXIS1'] == cat1.header['NAXIS1'] + cat2.header['NAXIS1']
        assert header['NAXIS2'] == cat1.header['NAXIS2']

        header = CatalogTools.create_new_header(cat1, cat2, naxis2='last')
        assert header['NAXIS2'] == cat2.header['NAXIS2']

        header = CatalogTools.create_new_header(cat1, cat2, naxis2='sum')
        assert header['NAXIS2'] == cat1.header['NAXIS2'] + cat2.header['NAXIS2']

    def test_flux_to_mag(self):
        flux = np.random.rand() * 100
        assert CatalogTools.flux_to_mag(flux) == -2.5*np.log10(flux) + 23.9
