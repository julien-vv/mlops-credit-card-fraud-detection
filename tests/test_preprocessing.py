import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocess_data import *

class TestPreprocessingData(unittest.TestCase):

    def create_raw_data(self):
        return pd.DataFrame({
            'Amount': [12, 23, 30],
            'Time': [1, 128989, 12876]
        })

    def test_hour_time_delete_for_scaled(self):
        # Given
        raw_data = self.create_raw_data()

        # When
        preprocessed = preprocessing_data(raw_data)

        # Then
        self.assertNotIn('Time', preprocessed.columns)
        self.assertNotIn('Hour', preprocessed.columns)
        self.assertIn('Amount_scaled', preprocessed.columns)
        self.assertIn('Hour_scaled', preprocessed.columns)

if __name__ == '__main__':
    unittest.main()
