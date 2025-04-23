# Import
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_XGB(X_train, y_train):
    # Addressing the imbalance problem using the scale pos weight method
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    model_XGB = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        eval_metric='logloss'
    )

    return model_XGB.fit(X_train, y_train)


def train_LR(X_train, y_train):
    model_LR = LogisticRegression(class_weight='balanced', random_state=42)
    return model_LR.fit(X_train, y_train)

def train_RF(X_train, y_train):
    model_RF = RandomForestClassifier(n_estimators=5, class_weight='balanced', random_state=42)
    return model_RF.fit(X_train, y_train)
