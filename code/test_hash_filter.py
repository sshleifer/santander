import unittest
from code.hash_filter import HashFilter
from code.constants import SMALL_DATA_PATH


class TestHashFilter(unittest.TestCase):

    def test_on_few_rows(self):
        hf = HashFilter(train_path=SMALL_DATA_PATH)
        self.assertGreaterEqual(200, len(hf.customers))
        hf.generate_submission(test_path=SMALL_DATA_PATH)
