import os
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump

def train_random_forest(X_train, y_train):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_res, y_res)

    dump(model, "models/random_forest_model.pkl")
    return model
