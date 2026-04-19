import numpy as np
import pandas as pd
import shap

from joblib import load
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

def get_model_specific_reason(feature, direction, model):
    if feature == "feature dimensionality":
        if direction == "high":
            if model == "gradient_boosting":
                return "high-dimensional data favors gradient boosting because it handles complex feature interactions"
            elif model == "random_forest":
                return "high-dimensional data favors random forests due to robustness to many features"
            elif model == "linear":
                return "high dimensionality can hurt linear models due to overfitting"
        else:  # low
            if model == "linear":
                return "low-dimensional data favors linear models due to simplicity"
            elif model == "random_forest":
                return "low-dimensional data reduces the need for complex ensembles like random forests"
            elif model == "gradient_boosting":
                return "low-dimensional data may not require complex boosting methods"

    elif feature == "feature interaction complexity":
        if direction == "high":
            if model == "gradient_boosting":
                return "gradient boosting captures complex feature interactions effectively"
            elif model == "random_forest":
                return "random forests can model non-linear interactions via tree splits"
            elif model == "linear":
                return "linear models struggle with complex feature interactions"
        else:
            if model == "linear":
                return "linear models perform well when interactions are simple"
            else:
                return "complex models are less necessary when interactions are simple"

    elif feature == "linearity of the dataset":
        if direction == "high":
            if model == "linear":
                return "linear models perform well on linear datasets"
            else:
                return ""
        else:
            if model in ["gradient_boosting", "random_forest"]:
                return "non-linear datasets favor tree-based models that capture complex patterns"
            else:
                return "linear models may struggle with non-linear relationships"

    return ""

# =========================
# RELOAD DATA (same as before)
# =========================
df = pd.read_csv(DATA_DIR / "tpot_results.csv")

df = df[df["status"] == "success"].copy()

counts = df["model_family"].value_counts()
rare_classes = counts[counts < 3].index
df["model_family_grouped"] = df["model_family"].replace(rare_classes, "other")

drop_cols = [
    "task_id", "dataset_name", "model", "pipeline",
    "model_family", "model_family_grouped", "status", "error",
    "hyperparameters", "max_depth", "n_estimators",
    "num_leaves", "cv_accuracy", "runtime_sec", "p_trace",
    "roy_root", "attr_ent.mean"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
X = X.select_dtypes(include=[np.number])
X = X.fillna(0)

# =========================
# LOAD SAVED TRANSFORMS
# =========================
vt = load(MODEL_DIR / "vt.joblib")
selector = load(MODEL_DIR / "selector.joblib")

model = load(MODEL_DIR / "randomForestModel.joblib")

rf_model = model

# =========================
# APPLY SAME TRANSFORMS
# =========================
X_vt = vt.transform(X)
X_selected = selector.transform(X_vt)

# Load background
X_background = np.load(
    MODEL_DIR / "shap_background.npy"
)

# Use ONLY the classifier
rf_model = model.named_steps["clf"]

explainer = shap.TreeExplainer(rf_model, data=X_background)

# Sample datasets
sample_idx = np.random.choice(len(X), 5, replace=False)
X_samples = X.iloc[sample_idx]

# Apply SAME transforms to samples
X_samples_vt = vt.transform(X_samples)
X_samples_selected = selector.transform(X_samples_vt)

# SHAP
shap_values = explainer.shap_values(X_samples_selected)

# Predictions MUST use transformed data
preds = model.predict(X_samples_selected)

# Feature names after transformations
vt_mask = vt.get_support()
X_vt_cols = X.columns[vt_mask]

selector_mask = selector.get_support()
selected_feature_names = X_vt_cols[selector_mask]

feature_names = selected_feature_names


explanations = []
theory_map = {
    "eq_num_attr": "supported",        # dimensionality → well-established
    "joint_ent.mean": "supported",     # interaction strength → core ML concept
    "lh_trace": "supported",           # linearity proxy → valid assumption
    "n_instances": "supported",        # dataset size → bias/variance theory
}

feature_name_map = {
    "eq_num_attr": "feature dimensionality",
    "joint_ent.mean": "feature interaction complexity",
    "lh_trace": "linearity of the dataset",
    "n_instances": "dataset size"
}

reasoning_map = {
    "feature dimensionality": "high-dimensional data favors flexible models like ensembles",
    "feature interaction complexity": "complex feature interactions are better captured by non-linear models like tree ensembles",
    "linearity of the dataset": "linear datasets favor simpler linear models",
    "dataset size": "larger datasets allow complex models to generalize better"
}

model_map = {
    "RandomForestClassifier": "random_forest",
    "GradientBoostingClassifier": "gradient_boosting",
    "LogisticRegression": "linear",
}

results = []

for i in range(len(X_samples)):
    pred_class = preds[i]
    le = load(MODEL_DIR / "label_encoder.joblib")
    algo_name = le.inverse_transform([pred_class])[0]
    algo_key = model_map.get(algo_name, algo_name)

    if algo_name == "other":
        continue
    
    # get SHAP values for predicted class
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_idx = pred_class
        sv = shap_values[class_idx, i]
    else:
        sv = shap_values[i]
    
    # get top features
    top_idx = np.argsort(np.abs(sv))[::-1][:3]
    
    explanation_parts = []
    
    for idx in top_idx:
        raw_feature = feature_names[idx]

        valid_features = set(feature_name_map.keys())

        if raw_feature not in valid_features:
            continue

        feature = feature_name_map.get(raw_feature, raw_feature)

        value = X_samples.iloc[i][raw_feature]
        impact = sv[idx]

        median_val = X[raw_feature].median()
        direction = "high" if value > median_val else "low"

        threshold = np.mean(np.abs(sv)) + np.std(np.abs(sv))
        strength = "strongly" if abs(impact) > threshold else "weakly"

        # ✅ NOW it's safe to use strength
        if raw_feature in theory_map and strength == "strongly":
            results.append(theory_map[raw_feature])

        # Direction-aware reasoning
        reason = get_model_specific_reason(feature, direction, algo_key)

        if not reason:
            reason = reasoning_map.get(feature, "")

        explanation_parts.append(
            f"{feature} was {direction} — this contributes {strength}"
            + (f", as {reason}" if reason else "")
        )

    if not explanation_parts:
            continue
      
    explanation = (
        f"Algorithm {algo_name} was selected because: "
        + ", ".join([f"({j+1}) {p}" for j, p in enumerate(explanation_parts)])
    )
    
    explanations.append(explanation)

for e in explanations:
    print("\n", e)

# scoring
supported_frac = sum(r == "supported" for r in results) / len(results)

print("Supported fraction:", supported_frac)