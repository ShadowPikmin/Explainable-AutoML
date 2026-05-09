import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import shap

from joblib import load
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

def interpret_feature(i, feature_name, feature_value, shap_value, X_full):
    
    # determine high/low using percentile rank
    col = X_full[feature_name]
    percentile = (col < feature_value).mean()

    if percentile > 0.66:
        value_level = "high"
    elif percentile < 0.33:
        value_level = "low"
    else:
        value_level = "medium"

    # strength based on SHAP magnitude
    abs_shap = abs(shap_value)

    if abs_shap > np.percentile(np.abs(X_full.values), 75):
        strength = "strongly"
    else:
        strength = "weakly"

    return value_level, strength


def explain_dataset(i, shap_vals, X_sample, X_full, pred_class_name):

    features = X_sample.columns
    values = X_sample.iloc[i].values

    parts = []

    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]

    for j in top_idx:
        feature = features[j]
        val = values[j]
        shap_val = shap_vals[j]

        # percentile-based interpretation
        col = X_full[feature]
        percentile = (col < val).mean()

        if percentile > 0.66:
            value_level = "high"
        elif percentile < 0.33:
            value_level = "low"
        else:
            value_level = "medium"

        abs_vals = np.abs(shap_vals)

        percentile_threshold = np.percentile(abs_vals, 75)
        absolute_threshold = np.mean(abs_vals)  # or std

        if abs(shap_val) >= percentile_threshold and abs(shap_val) > absolute_threshold:
            strength = "strongly"
        elif abs(shap_val) >= np.percentile(abs_vals, 40):
            strength = "moderately"
        else:
            strength = "weakly"

        direction = "contributes" if shap_val > 0 else "reduces"

        parts.append(
            f"{feature} was {value_level} — this {strength} {direction} toward selecting {pred_class_name}\n"
        )

    return f"Algorithm {pred_class_name} was selected because: " + ", ".join(parts) + "."

# =========================
# LOAD DATA (RECREATE X)
# =========================
df_tpot = pd.read_csv(DATA_DIR / 'tpot_results.csv')
df_metafeatures = pd.read_csv(DATA_DIR / 'metafeatures.csv')
model = load(MODEL_DIR / "randomForest.joblib")


#Sets up data 
#Groups classifier based on similar techniques
# Done to reduce algorithm only used once
family_map = {
    'LGBMClassifier': 'boosting',
    'XGBClassifier': 'boosting',
    'AdaBoostClassifier': 'boosting',

    'RandomForestClassifier': 'tree',
    'BaggingClassifier': 'tree',

    'LogisticRegression': 'linear',
    'SGDClassifier': 'linear',
    'LinearDiscriminantAnalysis': 'linear',

    'MLPClassifier': 'neural',

    'KNeighborsClassifier': 'instance',

    'QuadraticDiscriminantAnalysis': 'probabilistic',
    'BernoulliNB': 'probabilistic'
}

#Map algorithms from dataset to the family groups
y = df_tpot['algorithm'].map(family_map)

#Stores all metafeatures for each task 
X = df_metafeatures

#To keep track of task names 
meta = df_metafeatures[['task']].copy()

# Since instance and probablity only appear 1 or 2 times
# We shall remove them to improve model accuracy
valid_classes = ['boosting', 'linear', 'neural', 'tree']
mask = y.isin(valid_classes)

meta = meta[mask]
X = X[mask]
y = y[mask]

#Encodes the algorithm to numbers to improve training
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X.drop(['task'], axis = 1)

#Recreate the RF hyperparamters configuration 
selector = SelectKBest(f_classif, k=10)

#Preserves Feature Names 
X_selected = selector.fit_transform(X, y_encoded)
selected_features = X.columns[selector.get_support()]
X_selected = pd.DataFrame(
    selector.fit_transform(X, y_encoded),
    columns=X.columns[selector.get_support()],
    index=X.index 
)
  
X_sample = X_selected.sample(n=5, random_state=42)
X_sample_task = meta.loc[X_sample.index, 'task']
print("Data loaded")

X_train, X_test, y_train, t_test = train_test_split(X_selected,y_encoded, 
                                        random_state = 42, 
                                        test_size = 0.25,
                                        shuffle=True)

#Runs Shap TreeExplainer on RF model 
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

#Gets 5 local explanations
sample_predictions = model.predict(X_sample.values)
actual_algo = df_tpot.set_index('task').loc[
    X_sample_task, 'algorithm'
]

for i in range(5):

    pred_class = sample_predictions[i]
    pred_name = le.inverse_transform([pred_class])[0]

    shap_vals_for_sample = shap_values.values[i, :, pred_class]  # ✅ FIX

    explanation = explain_dataset(
        i,
        shap_vals_for_sample,
        X_sample,
        X,
        pred_name
    )

    print("\n" + "="*80)
    print("TASK:", X_sample_task.iloc[i])
    print(explanation)