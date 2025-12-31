import os
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from joblib import dump

def train_logistic(X_train, y_train):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_res, y_res)

    dump(model, "models/logistic_model.pkl")
    return model

