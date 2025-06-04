# Import
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    conf_mat = confusion_matrix(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred, output_dict=True)
    score_roc_auc = roc_auc_score(y_test, y_proba)
    auprc = round(average_precision_score(y_test, y_proba), 4)

    return conf_mat, class_rep, score_roc_auc, auprc