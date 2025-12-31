from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "AUC_PR": auc_pr,
        "F1": f1,
        "Confusion_Matrix": cm
    }
