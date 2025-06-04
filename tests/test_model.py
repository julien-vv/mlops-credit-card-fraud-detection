import sys
import os
import unittest
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from train_model import *

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Creation of fake data
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
        self.X_train = X
        self.y_train = y
        self.params = {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }

    def test_train_XGB(self):
        # Train XGB
        model_XGB = train_XGB(self.X_train, self.y_train, self.params)

        self.assertIsInstance(model_XGB, XGBClassifier)

        # Check if the model have 'feature_importances_'
        self.assertTrue(hasattr(model_XGB, 'feature_importances_'))

if __name__ == '__main__':
    unittest.main()
