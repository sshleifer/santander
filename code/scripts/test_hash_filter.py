import unittest
from scripts.turbo_clean import HashFilter


class TestHashFilter(unittest.TestCase):

    def test_on_few_rows(self):
        hf = HashFilter(n_rows=200)
        self.assertGreaterEqual(200, len(hf.customers))
