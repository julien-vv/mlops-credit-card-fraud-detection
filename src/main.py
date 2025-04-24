# Import
from preprocess_data import *
from evaluate_model import *
from train_model import *
import mlflow
import mlflow.sklearn

def main():
    with mlflow.start_run():

        path = '..\data\creditcard.csv'
        df = read_file(path)
        df = preprocessing_data(df)
        X_train, X_test, y_train, y_test = prep_for_training(df)

        # Train
        model1 = train_LR(X_train, y_train)
        model2 = train_RF(X_train, y_train)
        model3 = train_XGB(X_train, y_train)

        # Evaluate
        conf_mat1, class_rep1, score_roc_auc1, auprc1 = evaluate_model(model1, X_test, y_test)
        conf_mat2, class_rep2, score_roc_auc2, auprc2 = evaluate_model(model2, X_test, y_test)
        conf_mat3, class_rep3, score_roc_auc3, auprc3 = evaluate_model(model3, X_test, y_test)

        # Log with mlflow
        mlflow.log_metric("roc_auc_score_LR", score_roc_auc1)
        mlflow.log_metric("average_precision_LR", auprc1)

        mlflow.log_metric("roc_auc_score_RF", score_roc_auc2)
        mlflow.log_metric("average_precision_RF", auprc2)

        mlflow.log_metric("roc_auc_score_XGB", score_roc_auc3)
        mlflow.log_metric("average_precision_XGB", auprc3)

        mlflow.sklearn.log_model(model1, "LogisticRegression_model")
        mlflow.sklearn.log_model(model2, "RandomForest_model")
        mlflow.sklearn.log_model(model3, "XGBoost_model")

        print("Program done")

if __name__ == "__main__":
    main()
