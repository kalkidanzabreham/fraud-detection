import shap
import pandas as pd
import matplotlib.pyplot as plt

def plot_builtin_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(fi["feature"][:top_n][::-1], fi["importance"][:top_n][::-1])
    plt.title("Top Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return fi.head(top_n)


def shap_global_explanation(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values[1], X_train, plot_type="bar")
    shap.summary_plot(shap_values[1], X_train)

    return explainer, shap_values


def shap_force_plot(explainer, shap_values, X, index):
    shap.force_plot(
        explainer.expected_value[1],
        shap_values[1][index],
        X.iloc[index],
        matplotlib=True
    )
