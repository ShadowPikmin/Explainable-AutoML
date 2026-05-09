import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from itertools import product
from collections import Counter

from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import warnings

from sklearn.svm import LinearSVC
warnings.simplefilter("ignore", UserWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

#Gets datasets for classification training
df_tpot = pd.read_csv(DATA_DIR / 'tpot_results.csv')
df_metafeatures = pd.read_csv(DATA_DIR / 'metafeatures.csv')

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

#Utilizes cross-validation over LOOCV due to data only 
# being size 50 (too small for LOOCV)
cv = RepeatedStratifiedKFold(
    n_splits=3,
     n_repeats=10,
    random_state = 42
)

#Stores what were the features selected in each run 
selectors = [] 

#Stores the predicted and actual values of y, and the model 
# from each combination of hyerparamters tested
versions = {}

#Parameters that we will test for optimal RF
#Total Combination: 576 
n_estimators = [100, 200, 300]
max_depths = [None, 3, 5, 7]
min_sample_leaf = [2, 5 ,10] 
max_features = [None,"sqrt", 0.3, 0.5]
ks = [10, 15, 20, 25] #[3]

#Stores accuacy of each model version 
accuracy = [] 
count = 1

#Runs RF on all combinations and store results 
print("Running Random Forest Model")
for n_est, max_dep, max_feats, min_sample, kfold in product(*[n_estimators, max_depths, max_features, min_sample_leaf, ks]):
    y_true = []
    y_pred = []
    for train_ix, test_ix in cv.split(X, y_encoded):

        X_train_raw, X_test_raw = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y_encoded[train_ix], y_encoded[test_ix]

        selector = SelectKBest(f_classif, k=kfold)
        X_train= selector.fit_transform(X_train_raw, y_train)
        X_test = selector.transform(X_test_raw)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask]

        selectors.append(selected_features)

        model = RandomForestClassifier( n_estimators=n_est,
                max_depth=max_dep,
                min_samples_leaf=min_sample,
                max_features=max_feats,
                class_weight="balanced",
                random_state=42)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        y_true.extend(y_test)
        y_pred.extend(pred)

    versions[count] = [y_true, y_pred, model, n_est, max_dep, min_sample, max_feats, kfold]
    accuracy.append([balanced_accuracy_score(y_true, y_pred), accuracy_score(y_true, y_pred)])
    print(f"Finished Combination #{count}: {n_est} {max_dep} {max_feats} {min_sample} {kfold}")
    count += 1

vers = []

#Finds the best RF model to store for futher research 
bestVersion = None
bestAcc = [0,0]
for i in range(len(accuracy)):
    if bestAcc[0] < accuracy[i][0]:
        bestVersion = versions[i + 1]
        bestAcc = accuracy[i]
 
y_true_labels = le.inverse_transform(bestVersion[0])
y_pred_labels = le.inverse_transform(bestVersion[1])

#Prints a report on the results 
print(f'Combination: {bestVersion}')
print(confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_))
print(classification_report(
    y_true_labels,
    y_pred_labels,
    zero_division=0
))

dump(bestVersion[2], MODEL_DIR / 'randomForestSmall.joblib')
print("Random Forest Model completed \n")


#Runs a basic Logistic Regession model for comparison 
print("Runncing Logistic Regression")
y_true = []
y_pred = [] 
log_version = None

count = 1

for train_ix, test_ix in cv.split(X, y_encoded):

    X_train_raw, X_test_raw = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y_encoded[train_ix], y_encoded[test_ix]

    selector = SelectKBest(f_classif, k=3)
    X_train= selector.fit_transform(X_train_raw, y_train)
    X_test = selector.transform(X_test_raw)

    log_model = LogisticRegression(
        max_iter = 5000,
        class_weight='balanced'
    )

    log_model.fit(X_train, y_train)
    pred = log_model.predict(X_test)

    y_true.extend(y_test)
    y_pred.extend(pred)

log_version = [y_true, y_pred, log_model]
log_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f'Accuracy: {log_accuracy}')

y_true_labels = le.inverse_transform(y_true)
y_pred_labels = le.inverse_transform(y_pred)

print(confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_))
print(classification_report(
        y_true_labels,
        y_pred_labels,
        zero_division=0
))
dump(log_model, MODEL_DIR / "logisticRegression.joblib")
print("Logisitic Regression Model Completed")

#Runs a basic 
print("Runncing Linear SVC")
y_true = []
y_pred = [] 
lin_version = None

for train_ix, test_ix in cv.split(X, y_encoded):

    X_train_raw, X_test_raw = X.iloc[train_ix], X.iloc[test_ix]
    y_train, y_test = y_encoded[train_ix], y_encoded[test_ix]

    selector = SelectKBest(f_classif, k=3)
    X_train= selector.fit_transform(X_train_raw, y_train)
    X_test = selector.transform(X_test_raw)

    lin_model = LinearSVC(
        class_weight='balanced'
    )

    lin_model.fit(X_train, y_train)
    pred = lin_model.predict(X_test)

    y_true.extend(y_test)
    y_pred.extend(pred)

lin_version = [y_true, y_pred, log_model]
lin_accuracy = balanced_accuracy_score(y_true, y_pred)
print(f'Accuracy: {lin_accuracy}')

y_true_labels = le.inverse_transform(y_true)
y_pred_labels = le.inverse_transform(y_pred)

print(confusion_matrix(y_true_labels, y_pred_labels, labels=le.classes_))
print(classification_report(
        y_true_labels,
        y_pred_labels,
        zero_division=0
))
dump(lin_model, MODEL_DIR / 'linaerSVC.joblib')



## ---------   Visualization Code ---------- ##

#Accuracy Report 
proportion_report = y.value_counts(normalize = True)
proportions = proportion_report.values
baseline = max(proportions)

models = ['Baseline','Random Forest', 'Logisitic Regression', 'Linear SVC']
accs = [baseline, bestAcc[1], log_accuracy, lin_accuracy]

plt.barh(models, accs)

plt.title('Model Accuracy')

plt.axvline(x=baseline, color='red', linestyle='--', linewidth=2, label=f'baseline: {baseline}')

plt.ylabel('Model')
plt.yticks(fontsize=8)

plt.xlabel('Accuracy')
plt.legend()
plt.savefig(PNG_DIR / "RFAccuracy.png", dpi=300, bbox_inches="tight")
plt.show()

#Balanced Random Forest Visualization
balanced_baseline = 0.25

models = ['baseline', 'Random Forest']
accs = [balanced_baseline, bestAcc[0]]

plt.barh(models, accs)

plt.title('Model Accuracy')

plt.axvline(x=balanced_baseline, color='red', linestyle='--', linewidth=2, label=f'baseline: {balanced_baseline}')

plt.ylabel('Model')
plt.yticks(fontsize=8)

plt.xlabel('Accuracy')
plt.legend()
plt.savefig(PNG_DIR / "BalancedRFAccuracy.png", dpi=300, bbox_inches="tight")
plt.show()

confusion_matrix = metrics.confusion_matrix(le.inverse_transform(bestVersion[0]), le.inverse_transform(bestVersion[1]))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix= confusion_matrix, display_labels = y.unique().tolist())

cm_display.plot()
plt.savefig(PNG_DIR / "RFconfusionMatrix.png", dpi=300, bbox_inches="tight")
plt.show()

selector_counts = Counter()

for s in selectors:
    selector_counts.update(s)

# Get top 10 most selected features
top_10 = selector_counts.most_common(10)

# Separate names and counts
features = [x[0] for x in top_10]
counts = [x[1] for x in top_10]

# Plot
plt.figure(figsize=(10, 6))

plt.barh(features, counts)

plt.xlabel('Selection Count')
plt.ylabel('Meta-feature')
plt.title('Top 10 Most Selected Meta-features')

plt.gca().invert_yaxis()  # largest on top

plt.tight_layout()
plt.savefig(PNG_DIR / "Top10Meta-features.png", dpi=300, bbox_inches="tight")
plt.show()
