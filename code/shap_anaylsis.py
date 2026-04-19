import shap 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import load
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"
# =========================
# LOAD DATA (RECREATE X)
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

# =========================
# LOAD MODEL
# =========================
model = load(MODEL_DIR / "randomForestModel.joblib")
vt = load(MODEL_DIR / "vt.joblib")
selector = load(MODEL_DIR / "selector.joblib")

rf_model = model.named_steps["clf"]

# =========================
# APPLY SAME TRANSFORMS
# =========================
# 1. Fill NaNs (same as training)
X = X.fillna(0)

# 2. VarianceThreshold (CRITICAL FIX)
X_vt = vt.transform(X)

# 3. SelectKBest
X_selected = selector.transform(X_vt)

np.save(
    MODEL_DIR / "shap_background.npy",
    X_selected
)
# =========================
# GET FEATURE NAMES
# =========================
# After variance threshold
vt_mask = vt.get_support()
X_vt_columns = X.columns[vt_mask]

# After SelectKBest
selector_mask = selector.get_support()
selected_feature_names = X_vt_columns[selector_mask]

# =========================
# SHAP
# =========================
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_selected)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# =========================
# HANDLE MULTI-CLASS SHAP
# =========================
shap_vals = shap_values

# Case 1: list (old multiclass API)
if isinstance(shap_vals, list):
    shap_vals = np.array(shap_vals)  # (classes, samples, features)
    shap_vals = np.mean(np.abs(shap_vals), axis=0)

# Case 2: 3D array (new API: samples, features, classes)
elif len(shap_vals.shape) == 3:
    shap_vals = np.mean(np.abs(shap_vals), axis=2)

# Case 3: already 2D → just take abs
else:
    shap_vals = np.abs(shap_vals)

# FINAL SAFETY CHECK
print("SHAP shape:", shap_vals.shape)

# =========================
# BEESWARM PLOT
# =========================
os.makedirs(PNG_DIR, exist_ok=True)

plt.figure()

shap.summary_plot(
    shap_vals,
    X_selected,
    feature_names=selected_feature_names,
    show=False
)

plt.tight_layout()
plt.savefig(PNG_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================
# GLOBAL FEATURE IMPORTANCE
# =========================
# Mean absolute SHAP per feature
mean_shap = np.mean(shap_vals, axis=0)

importance_df = pd.DataFrame({
    "feature": selected_feature_names,
    "importance": mean_shap
}).sort_values(by="importance", ascending=False)

# Save full importance (optional but useful for research)
importance_df.to_csv(
    DATA_DIR / "shap_importance.csv",
    index=False
)

# =========================
# TOP 10 BAR PLOT
# =========================
top_k = 10
top_features = importance_df.head(top_k)

plt.figure()

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 Most Important Meta-Features")

plt.tight_layout()
plt.savefig(
    PNG_DIR / "shap_top10_bar.png",
    dpi=300
)
plt.close()

print("\nTop 10 Features:")
print(top_features)

