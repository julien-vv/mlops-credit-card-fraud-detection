# Import
from preprocess_data import *
from evaluate_model import *
from train_model import *
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri('http://127.0.0.1:5000')

def main():
    with mlflow.start_run():

        path = '/app/data/creditcard.csv'
        df = read_file(path)
        df = preprocessing_data(df)
        X_train, X_test, y_train, y_test = prep_for_training(df)
        # Train
        model_type = 'LR'
        model_type = 'RF'
        model_type = 'XGB'
        model = train_model(X_train, y_train, model_type=model_type)

        # Evaluate
        conf_mat, class_rep, score_roc_auc, auprc = evaluate_model(model, X_test, y_test)

        # Log with mlflow
        mlflow.log_metric("roc_auc_score", score_roc_auc)
        mlflow.log_metric("average_precision", auprc)

        mlflow.set_tag("model_type", model_type)
        mlflow.sklearn.log_model(model, "model")

        print("Program done")

if __name__ == "__main__":
    main()