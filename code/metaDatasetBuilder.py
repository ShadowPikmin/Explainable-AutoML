import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os, time
warnings.filterwarnings('ignore')

from scipy.stats import rankdata
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Meta-learner
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from pymfe.mfe import MFE

# AutoML 
from sklearn.model_selection import KFold
from tpot import TPOTClassifier


#dataset
import openml
openml.config.apikey = '' 

#Storing Data
from pathlib import Path


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#Gets root directory
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"


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
            print(f'loaded {name} from OpenML: {X[mask].shape}')

        except Exception as e:
            print(f'  OpenML {name} failed ({e}) — using synthetic fallback')
            X = np.random.randn(200, 10)
            y = np.random.randint(0, 3, 200)
            datasets.append((name+'_synth', X, y))
        
    return datasets

def meta_feature_extraction(dataset):
    _, X, y = dataset
    mfe = MFE(groups=["statistical", "info-theory", "landmarking"]) 
    mfe.fit(X,y)
    names, values = mfe.extract()
    ft = dict(zip(names, values))

    #Cleans up data by turning all NaN and other undefined values 0
    #Allows us to store data as float values 
    for k in ft: 
        if ft[k] is None or not np.isfinite(ft[k]):
            ft[k] = 0.0

    return ft

def tpot_process(dataset):
    name, X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
    
    #The amount of kFold for TPoT
    k = 3
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    #TPOT model 
    model = TPOTClassifier(scorers=['accuracy'], verbose=2, max_time_mins=5, random_state=42, cv=kf, n_jobs=5)
    
    start_time = time.time()
    model.fit(X_train,y_train)
    total_time = time.time() - start_time

    #Uses the model trained in TPOT and cross validates its with the testing data
    cv_score = model.fitted_pipeline_.score(X_test, y_test)
    algo_name = type(model.fitted_pipeline_.steps[-1][1]).__name__ 

    #Returns the task name, accuracy score, and runtime 
    return {"task": name, 
            "algorithm": algo_name,
            "cv_accuracy": cv_score,
            "runtime": total_time}

def main(): 
    #load datasets
    datasets = load_dataset()
    print("\n\n")

    #Stores all metafeatures and TPotResutls
    metafeatures_results = []
    tpot_results = []

    #For each dataset extracts its metafeatuers and trains a TPOT model 
    # on it storing both results in respective arrays for file transfer
    count = 1
    for data in datasets: 
        name, _, _ = data
        
        print(f"Working on dataset #{count}: {name}")
        metafeatures = meta_feature_extraction(data)
        metafeatures_result = {"task":name, **metafeatures}
        metafeatures_results.append(metafeatures_result)
        
        tpot_results.append(tpot_process(data))
        print(f"Dataset {name} completed\n")
        count += 1

    pd.DataFrame(tpot_results).to_csv(DATA_DIR / "tpot_results.csv", index=False)
    pd.DataFrame(metafeatures_results).to_csv(DATA_DIR / "metafeatures.csv", index=False)
        
    print("Data saved to dataset folder")
        


if __name__ == "__main__":
    main()