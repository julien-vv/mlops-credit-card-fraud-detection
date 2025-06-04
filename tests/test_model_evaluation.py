import sys
import os
import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluate_model import *

class TestEvaluateModel(unittest.TestCase):

    def setUp(self):
        # Fake dataset
        self.X_test = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])  # Exemple de features
        self.y_test = np.array([0, 1, 1, 0])  # Labels réels (0 ou 1)

        # Train a model with this data
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_test, self.y_test)  # Entraîner le modèle sur les données factices

    def test_evaluate_model(self):
        conf_mat, class_rep, score_roc_auc, auprc = evaluate_model(self.model, self.X_test, self.y_test)

        self.assertEqual(conf_mat.shape, (2, 2), "Confusion matric should have a shape (2, 2)")

        self.assertIn('precision', class_rep['0'])
        self.assertIn('recall', class_rep['0'])
        self.assertIn('precision', class_rep['1'])
        self.assertIn('recall', class_rep['1'])
        self.assertIn('precision', class_rep['macro avg'])
        self.assertIn('recall', class_rep['macro avg'])
        self.assertIn('precision', class_rep['weighted avg'])
        self.assertIn('recall', class_rep['weighted avg'])

        self.assertGreaterEqual(score_roc_auc, 0, "Le score ROC AUC doit être supérieur ou égal à 0")
        self.assertLessEqual(score_roc_auc, 1, "Le score ROC AUC doit être inférieur ou égal à 1")

        self.assertGreaterEqual(auprc, 0, "Le AUPRC doit être supérieur ou égal à 0")
        self.assertLessEqual(auprc, 1, "Le AUPRC doit être inférieur ou égal à 1")

if __name__ == "__main__":
    unittest.main()
