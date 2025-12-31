import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

def stratified_cv(model, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    f1_scores = []
    auc_pr_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        f1_scores.append(f1_score(y_test, y_pred))
        auc_pr_scores.append(auc(recall, precision))

    return {
        "F1_mean": np.mean(f1_scores),
        "F1_std": np.std(f1_scores),
        "AUC_PR_mean": np.mean(auc_pr_scores),
        "AUC_PR_std": np.std(auc_pr_scores)
    }
