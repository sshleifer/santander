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
        preds = sf.predict_each_row(df)
        vset = sf.make_validation_set(df.tail(50)).to_frame(name='truth')
        map7 = sf.score(preds, vset)  # all products have already been added
        self.assertEqual(map7, 0)
        preds.to_csv('submissions/test.csv')
        with self.assertRaises(AssertionError):
            is_valid_submission('submissions/test.csv')


    def test_that_hash_cols_are_fixed(self):
        # df=store['df_train']
        # df.
        pass
