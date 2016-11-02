import pandas as pd
import unittest

from code.hash_filter import HashFilter, is_valid_submission, SamFilter
from code.constants import SMALL_DATA_PATH


class TestHashFilter(unittest.TestCase):

    def test_on_few_rows(self):
        hf = HashFilter(train_path=SMALL_DATA_PATH)
        self.assertGreaterEqual(200, len(hf.customers))
        hf.generate_submission(test_path=SMALL_DATA_PATH)

    def test_sam_filter(self):
        df = pd.read_csv(SMALL_DATA_PATH)
        sf = SamFilter(df)
        self.assertGreaterEqual(200, len(sf.customer_usage))
        self.preds = sf.predict_each_row(df)
        self.preds.to_csv('submissions/test.csv')
        self.assertFalse(is_valid_submission('submissions/test.csv'))
