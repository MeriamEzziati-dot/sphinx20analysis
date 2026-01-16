import numpy as np

from MER_analysis.catalogs import MatchedCatalog


class TestMatchedCatalog:
    def test_from_catalogs(self, dummy_catalog):
        cat1 = dummy_catalog
        cat1.extract_coords()
        cat2 = dummy_catalog
        cat2.extract_coords()
        catalog = MatchedCatalog.from_catalogs(cat1, cat2)
        assert type(catalog) == MatchedCatalog
        np.testing.assert_array_equal(catalog.data['dist_match'], np.zeros_like(catalog.data['dist_match']))



