from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import shap 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import dump, load
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"
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
print(df_tpot['algorithm'].value_counts())

#Stores all metafeatures for each task 
X = df_metafeatures.drop(['task'], axis = 1)

# Since instance and probablity only appear 1 or 2 times
# We shall remove them to improve model accuracy
valid_classes = ['boosting', 'linear', 'neural', 'tree']
mask = y.isin(valid_classes)

X = X[mask]
y = y[mask]

#Encodes the algorithm to numbers to improve training
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#Recreate the RF hyperparamters configuration 
selector = SelectKBest(f_classif, k=10)

#Preserves Feature Names 
X_selected = selector.fit_transform(X, y_encoded)
selected_features = X.columns[selector.get_support()]
X_selected = pd.DataFrame(
    X_selected,
    columns=selected_features
)

print("Data loaded")

X_train, X_test, y_train, t_test = train_test_split(X_selected,y_encoded, 
                                        random_state = 42, 
                                        test_size = 0.25,
                                        shuffle=True)

#Runs Shap TreeExplainer on RF model 
explainer = shap.TreeExplainer(model)
#Gets shap_values for datasets 
shap_values = explainer.shap_values(X_test)

print(f"SHAP values calculated for {shap_values.shape[0]} samples")
print(f"Each prediction explained by {shap_values.shape[1]} selected features")
print(f"The number of classes were {shap_values.shape[2]}")

# The base value (expected value) - what the model predicts on "average"
print("This is reached")
print(f"Model's base prediction (expected value): ${explainer.expected_value}")
print("Maybe")
# Quick verification: SHAP values should be additive
sample_idx = 0

#Gets the predicted probabilty of each family 
pred_probs = model.predict_proba(X_test)[sample_idx]

print("Predicted probabilities:")
print(pred_probs)

#Report on the SHAP values for each family 
for class_idx in range(len(le.classes_)):

    shap_sum = (
        explainer.expected_value[class_idx]
        + np.sum(shap_values[sample_idx, :, class_idx])
    )

    print(f"\nClass: {le.classes_[class_idx]}")
    print(f"Model prob: {pred_probs[class_idx]:.4f}")
    print(f"SHAP sum : {shap_sum:.4f}")

#Calculates the absolute shap mean value 
mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
print(f'The mean. absolute SHAP values across all datasets is: {mean_abs_shap}')

#Cretaes a Dataframe that summarizes glabal feature importance
# through calculating the avg abs SHAP value for each feature
selected_features = X.columns[selector.get_support()]
shap_importance = pd.DataFrame({
    'feature': selected_features,
    'mean_abs_shap': mean_abs_shap
})

#Sorts global features importance by avg abs SHAP values
shap_importance = shap_importance.sort_values(
    by='mean_abs_shap',
    ascending=False
)

print(shap_importance)

#Plots the global features 
plt.figure(figsize=(8,5))

plt.barh(
    shap_importance['feature'],
    shap_importance['mean_abs_shap']
)

plt.xlabel('Mean Absolute SHAP Value')
plt.ylabel('Meta-feature')
plt.title('Global SHAP Feature Importance')

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ===== SHAP beeswarm matrix =====
shap_beeswarm = np.mean(
    np.abs(shap_values),
    axis=2
)

# ===== Beeswarm Plot =====
shap.summary_plot(
    shap_beeswarm,
    X_test,
    feature_names=selected_features,
    show=False
)

plt.title("SHAP Beeswarm Plot")
plt.tight_layout()

plt.savefig(
    PNG_DIR / "shap_beeswarm.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

#Plots Bar graph 
top10 = shap_importance.head(10)

plt.figure(figsize=(8,5))

plt.barh(
    top10['feature'],
    top10['mean_abs_shap']
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Meta-feature")
plt.title("Top SHAP Meta-feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    PNG_DIR / "shap_barplot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()