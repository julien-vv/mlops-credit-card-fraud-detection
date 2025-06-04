from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def train_XGB(X_train, y_train, params):
    # Addressing the imbalance problem using the scale pos weight method
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    model_xgb = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight
    )
    model = model_xgb.fit(X_train, y_train)

    return model


def train_LR(X_train, y_train, params):
    model_lr = LogisticRegression(**params)
    model = model_lr.fit(X_train, y_train)

    return model


def train_RF(X_train, y_train, params):
    model_rf = RandomForestClassifier(**params)
    model = model_rf.fit(X_train, y_train)

    return model


def train_model(X_train, y_train, model_type, params):
    if model_type == 'XGB':
        print("Training XGBoost model...")
        return train_XGB(X_train, y_train, params)
    elif model_type == 'LR':
        print("Training Logistic Regression model...")
        return train_LR(X_train, y_train, params)
    elif model_type == 'RF':
        print("Training Random Forest model...")
        return train_RF(X_train, y_train, params)
    else:
        raise ValueError("Invalid model type. Choose from 'XGB', 'LR', or 'RF'.")
