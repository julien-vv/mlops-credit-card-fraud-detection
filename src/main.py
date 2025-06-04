# Import
from preprocess_data import *
from evaluate_model import *
from train_model import *
import mlflow

## HELP ##
# mlflow ui

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("Fraud Detection")


def main():

    # Preprocessing
    path = '../data/creditcard.csv'
    df = read_file(path)
    df = preprocessing_data(df)
    X_train, X_test, y_train, y_test = prep_for_training(df)

    # Parameters of the model
    params_xgb = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    params_lr = {
        'class_weight': 'balanced',
        'random_state': 42
    }
    params_rf = {
        'n_estimators': 5,
        'class_weight': 'balanced',
        'random_state': 42
    }
    models = [
        (
            "XGB",
            params_xgb
        ),
        (
            "LR",
            params_lr
        ),
        (
            "RF",
            params_rf
        )
    ]

    # Train
    for model_type, params in models:
        model = train_model(X_train, y_train, model_type, params)

        # Evaluate
        conf_mat, class_rep, score_roc_auc, auprc = evaluate_model(model, X_test, y_test)

        # Use of MLFlow
        with mlflow.start_run(run_name=model_type) as run:
            mlflow.log_param('model', model_type)
            mlflow.log_params(params)
            mlflow.log_metrics({
                'accuracy': class_rep['accuracy'],
                'recall_class_0': class_rep['0']['recall'],
                'recall_class_1': class_rep['1']['recall'],
                'f1_score_macro': class_rep['macro avg']['f1-score']
            })
            if model_type == "XGB":
                mlflow.xgboost.log_model(model, "model")
                model.save_model("model.json")
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, model_type)
            else:
                mlflow.sklearn.log_model(model, "model")

    print("Program done")

if __name__ == "__main__":
    main()