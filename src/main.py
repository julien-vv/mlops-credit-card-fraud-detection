# Import
from preprocess_data import *
from evaluate_model import *
from train_model import *

def main():
    # Take the path of the dataset
    path = '..\data\creditcard.csv'
    # Read the file
    df = read_file(path)
    # Preprocess the dataset
    df = preprocessing_data(df)
    # Split the dataset
    X_train, X_test, y_train, y_test = prep_for_training(df)

    # Train 3 different model
    model1 = train_LR(X_train, y_train)
    model2 = train_RF(X_train, y_train)
    model3 = train_XGB(X_train, y_train)

    # Evaluate 3 different model
    conf_mat1, class_rep1, score_roc_auc1, auprc1 = evaluate_model(model1, X_test, y_test)
    conf_mat2, class_rep2, score_roc_auc2, auprc2 = evaluate_model(model2, X_test, y_test)
    conf_mat3, class_rep3, score_roc_auc3, auprc3 = evaluate_model(model3, X_test, y_test)

    print("Program done")

if __name__ == "__main__":
    main()
