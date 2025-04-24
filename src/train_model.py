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
    model = model_XGB.fit(X_train, y_train)

    return model


def train_LR(X_train, y_train):
    model_LR = LogisticRegression(class_weight='balanced', random_state=42)
    model = model_LR.fit(X_train, y_train)

    return model


def train_RF(X_train, y_train):
    model_RF = RandomForestClassifier(n_estimators=5, class_weight='balanced', random_state=42)
    model = model_RF.fit(X_train, y_train)

    return model


def train_model(X_train, y_train, model_type='XGB'):
    """
    Chooses the model to train based on the user's input.

    Parameters:
    X_train (array-like): Features for training.
    y_train (array-like): Target labels for training.
    model_type (str): Type of model to train. Options are 'XGB', 'LR', 'RF'. Default is 'XGB'.

    Returns:
    model: Trained model.
    """
    if model_type == 'XGB':
        print("Training XGBoost model...")
        return train_XGB(X_train, y_train)
    elif model_type == 'LR':
        print("Training Logistic Regression model...")
        return train_LR(X_train, y_train)
    elif model_type == 'RF':
        print("Training Random Forest model...")
        return train_RF(X_train, y_train)
    else:
        raise ValueError("Invalid model type. Choose from 'XGB', 'LR', or 'RF'.")
