import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_DIR / "tpot_results.csv")

# -------------------------
# Choose label
# -------------------------
label_col = "model_family" # or "model_family_grouped"

# Drop failed runs if any
df = df[df["status"] == "success"].copy()
counts = df["model_family"].value_counts()

threshold = 4
rare_classes = counts[counts < threshold].index
df["model_family_grouped"] = df["model_family"].replace(rare_classes, "other")

# =========================
# PREPARE FEATURES
# =========================
# Remove non-meta columns
drop_cols = [
    "task_id", "dataset_name", "model", "pipeline",
    "model_family", "model_family_grouped", "status", "error",
    "hyperparameters", "max_depth", "n_estimators",
    "num_leaves","cv_accuracy", "runtime_sec", "p_trace",
    "roy_root", "attr_ent.mean"
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
X = X.select_dtypes(include=[np.number])
y = df[label_col]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
dump(le, MODEL_DIR / "label_encoder.joblib")

# Fill NaNs BEFORE feature selection
X = X.fillna(0)
vt = VarianceThreshold(threshold=0.01)
X = vt.fit_transform(X)
dump(vt, MODEL_DIR / "vt.joblib")


# =========================
# NAIVE BASELINE
# =========================
baseline = y.value_counts(normalize=True).max()
print(f"\nNaive baseline accuracy: {baseline:.3f}")

# =========================
# LEAVE-ONE-OUT CV
# =========================
loo = LeaveOneOut()

y_true = []
y_pred = []

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y_encoded)
dump(selector, MODEL_DIR / "selector.joblib")

for train_idx, test_idx in loo.split(X_selected):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    y_true.append(y_test[0])
    y_pred.append(pred[0])

    print(classification_report(
        y_true,
        y_pred,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0
    ))
dump(model, MODEL_DIR / "randomForestModel.joblib")
print("Model saved")

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average=None)
f1_macro = f1_score(y_true, y_pred, average="macro")

print(f"\nMeta-model accuracy: {accuracy:.3f}")

#f1_marco
print("n\ F1 Macro")
print(f"f1 Macro: {f1_macro:.3f}")

# Per-class F1
print("\nPer-class F1 scores:")
for i, score in enumerate(f1):
    print(f"{le.inverse_transform([i])[0]}: {score:.3f}")

# Full report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =========================
# COMPARE TO BASELINE
# =========================
if accuracy > baseline:
    print("\nMeta-model beats the naive baseline")
else:
    print("\n Meta-model does not beat the baseline")

plt.figure()
im = plt.imshow(cm)
plt.colorbar(im)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.xticks(ticks=range(len(le.classes_)), labels=le.classes_, rotation=45)
plt.yticks(ticks=range(len(le.classes_)), labels=le.classes_)

plt.tight_layout()
plt.savefig(PNG_DIR / "confusion_matrix.png", dpi=300)
plt.show()
