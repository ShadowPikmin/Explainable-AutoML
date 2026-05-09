import time
from typing import Counter

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import shap 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openml
import warnings
warnings.filterwarnings("ignore")


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

TIME_LIMIT_SECONDS = 300   # 5 minutes
RANDOM_STATE = 42

FAMILY_MODELS = {
    "boosting": ["LGBMClassifier", "XGBClassifier", "AdaBoostClassifier"],
    "tree": ["RandomForestClassifier", "BaggingClassifier"],
    "linear": ["LogisticRegression", "SGDClassifier", "LinearDiscriminantAnalysis"],
    "neural": ["MLPClassifier"],
    "instance": ["KNeighborsClassifier"],
    "probabilistic": ["QuadraticDiscriminantAnalysis", "BernoulliNB"]
}

def load_dataset():
    #Gets datasets from openml suite
    suite = openml.study.get_suite("OpenML-CC18")
    tasks_ids = list(suite.tasks)
    tasks_ids = tasks_ids[:50]

    #Loops through each dataset and preprocesses 
    #them for tpot and metafeature extraction
    datasets = []
    for task_id in tasks_ids: 
        task= openml.tasks.get_task(task_id)
        ds = task.get_dataset()
        name = ds.name
        try: 
            X,y,_,_ = ds.get_data(target = ds.default_target_attribute)
            X = pd.get_dummies(X, dummy_na=True).fillna(0)
            X = np.array(X, dtype=float)
            y = LabelEncoder().fit_transform(y)

            #Remove NaN
            mask = ~np.isnan(X).any(axis = 1)

            datasets.append((name, X[mask], y[mask]))

        except Exception as e:
            X = np.random.randn(200, 10)
            y = np.random.randint(0, 3, 200)
            datasets.append((name+'_synth', X, y))
        
    return datasets

def evaluate_model_cv(
    dataset,
    model_name,
    cv=3,
    scoring="accuracy",
    random_state=42
):
    """
    Evaluate a classifier using cross-validation.

    Parameters
    ----------
    dataset : tuple
        (dataset_name, X, y)

    model_name : str
        Name of classifier.

    cv : int
        Number of CV folds.

    scoring : str
        Scoring metric.

    random_state : int
        Random seed.

    Returns
    -------
    mean_score, std_score, scores
    """

    _, X, y = dataset

    # =========================
    # MODEL SELECTION
    # =========================
    models = {

        # Boosting
        "LGBMClassifier": LGBMClassifier(
            random_state=random_state,
            verbose=-1
        ),

        "XGBClassifier": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            n_estimators=50,
            tree_method="hist",
            n_jobs=-1
        ),

        "AdaBoostClassifier": AdaBoostClassifier(
            random_state=random_state
        ),

        # Tree
        "RandomForestClassifier": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        ),

        "BaggingClassifier": BaggingClassifier(
            random_state=random_state,
            n_jobs=-1
        ),

        # Linear
        "LogisticRegression": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            n_jobs=-1
        ),

        "SGDClassifier": SGDClassifier(
            random_state=random_state
        ),

        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),

        # Neural
        "MLPClassifier": MLPClassifier(
            random_state=random_state,
            max_iter=500
        ),

        # Instance
        "KNeighborsClassifier": KNeighborsClassifier(
            n_jobs=-1
        ),

        # Probabilistic
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),

        "BernoulliNB": BernoulliNB()
    }

    # =========================
    # GET MODEL
    # =========================
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    model = models[model_name]

    # =========================
    # CROSS VALIDATION
    # =========================
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        n_jobs=-1
    )

    return np.mean(scores), np.std(scores), scores

def evaluate_family(dataset, family, cv=3):
    _, X, y = dataset

    best_score = -1
    best_model = None

    for model_name in FAMILY_MODELS[family]:

        mean, _, _ = evaluate_model_cv(
            dataset,
            model_name,
            cv=cv
        )

        if mean > best_score:
            best_score = mean
            best_model = model_name

    return best_model, best_score

def evaluate_all_families(dataset):
    results = {}

    for family in FAMILY_MODELS.keys():
        best_model, best_score = evaluate_family(dataset, family)
        results[family] = (best_model, best_score)

    return results


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

#Gets the predicted probabilty of each family 
pred_probs = model.predict(X_selected)

tpot_data = df_tpot[['task','algorithm']]
tpot_data['algorithm'] = tpot_data['algorithm'].map(family_map)

pred_probs = le.inverse_transform(pred_probs)


count = 0
index = 0
missed_task = []
for i in range(len(pred_probs)):
    task = tpot_data['task'][i]
    pred = pred_probs[i]
    actual_algo = tpot_data['algorithm'][i]

    if(pred != actual_algo):
        count += 1
        missed_task.append([index, pred])

    index += 1
    
print(f"RF choose a different algorithm family {count} times\n\n")

# List all tasks as a dataframe
#load datasets
datasets = load_dataset()

print("reached")
cv_accuracy = {"Meta-Model": 0, "Both": 0, "TPOT": 0}
model_loss = []
model_won = []
for index, family in missed_task:
    task = tpot_data['task'][index]
    filtered_tasks = datasets[index]
    
    best_model, best_score = evaluate_family(filtered_tasks, family, cv=3)
    tpot_cv = df_tpot['cv_accuracy'][index]
    if(best_score > tpot_cv):
        print(f'{best_model} had a better performance than TPOT with an accuracy of {best_score} compared to {tpot_cv}')
        cv_accuracy["Meta-Model"] += 1
        model_won.append(best_model)
    elif(best_score < tpot_cv):
        cv_accuracy["TPOT"] +=1 
        print(f'{best_model} had a worser performance than TPOT with an accuracy of {best_score} compared to {tpot_cv}')
        model_loss.append(best_model)
    else:
        cv_accuracy["Both"] += 1
        print(f'{best_model} performed as well as TPOT with an accuracy of {best_score}')

#Difference Report
meta = cv_accuracy['Meta-Model']
tpot = cv_accuracy['TPOT']
both = cv_accuracy['Both']

labels = ['RF', 'TPOT', 'BOTH']
frequency = [meta, tpot, both]

plt.barh(labels, frequency)
plt.title('Random Forest vs TPOT on Disagreement')

plt.ylabel('Winners')
plt.yticks(fontsize=12)

plt.xlabel('Frequency')
plt.savefig(PNG_DIR / "RFvTPOTDisagreement.png", dpi=300, bbox_inches="tight")
plt.show()

#Model Won report
won_family = [family_map[m] for m in model_won]
counts = Counter(won_family)

families = list(counts.keys())
frequency = list(counts.values())

total = sum(frequency)
frequency_pct = [f / total * 100 for f in frequency]


plt.barh(families, frequency_pct)
plt.title('Meta-Model vs TPOT: Win Frequency by Family')

plt.ylabel('Algorithm Family')
plt.yticks(fontsize=12)

plt.xlabel('Frequency (Proportions)')
plt.savefig(PNG_DIR / "RFvTPOTwin.png", dpi=300, bbox_inches="tight")
plt.show()

#Model Loss report
loss_family = [family_map[m] for m in model_loss]
counts = Counter(loss_family)

loss_families = list(counts.keys())
frequency = list(counts.values())

total = sum(frequency)
frequency_pct = [f / total * 100 for f in frequency]


plt.barh(loss_families, frequency_pct)
plt.title('Meta-Model vs TPOT: Loss Frequency by Family')

plt.ylabel('Algorithm Family')
plt.yticks(fontsize=12)

plt.xlabel('Frequency (Proportions)')
plt.savefig(PNG_DIR / "RFvTPOTloss.png", dpi=300, bbox_inches="tight")
plt.show()