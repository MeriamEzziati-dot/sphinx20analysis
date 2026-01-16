from MER_analysis.catalogs import MERCatalog


class TestMERCatalog:
    def test_from_file(self, mer_catalog_fits_file):
        mer_catalog = MERCatalog.from_file(mer_catalog_fits_file)
        assert type(mer_catalog) == MERCatalog
        # TODO: improve test

    def test_fix_columns_names(self, mer_catalog_fits_file):
        mer_catalog = MERCatalog.from_file(mer_catalog_fits_file)
        assert set(MERCatalog.NEW_COLUMNS).issubset(set(mer_catalog.columns))
        # TODO: improve test
