import openml
import pandas as pd
import numpy as np
import time
import warnings
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tpot import TPOTClassifier
from tqdm import tqdm
from pymfe.mfe import MFE
from openml.tasks import TaskType
from pathlib import Path


from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

#Gets root directory
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

# kill Dask dashboard noise
os.environ["DASK_DISTRIBUTED__DASHBOARD__ENABLED"] = "false"

# silence sklearn + lightgbm feature warnings
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

RANDOM_SEED = 42
TIME_LIMIT_MINS = 5

MAX_WORKERS = min(2, os.cpu_count() - 1)   # 🔥 SAFE for Mac (increase to 3–4 if M-series Pro/Max)
CACHE_DIR = "openml_cache"
OUTPUT_PATH = DATA_DIR / "tpot_results.csv"
openml.config.cache_directory = CACHE_DIR

def clean_params(params):
    clean = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            clean[k] = v
        except:
            clean[k] = str(v)
    return clean

def get_model_family(name):
    if "LGBM" in name:
        return "gradient_boosting"
    elif "RandomForest" in name:
        return "random_forest"
    elif "LogisticRegression" in name:
        return "linear"
    else:
        return "other"
    

def compute_landmarks(X, y):
    results = {}

    try:
        results["landmark_1nn"] = np.mean(cross_val_score(
            KNeighborsClassifier(n_neighbors=1), X, y, cv=3
        ))
    except:
        results["landmark_1nn"] = 0.0

    try:
        results["landmark_nb"] = np.mean(cross_val_score(
            GaussianNB(), X, y, cv=3
        ))
    except:
        results["landmark_nb"] = 0.0

    try:
        results["landmark_tree"] = np.mean(cross_val_score(
            DecisionTreeClassifier(max_depth=1), X, y, cv=3
        ))
    except:
        results["landmark_tree"] = 0.0

    return results
    
def extract_meta_features(X, y):
    try:
        X = pd.DataFrame(X).copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # encode categoricals
            for col in X.select_dtypes(include=["object", "category"]):
                X[col] = X[col].astype("category").cat.codes

            # remove constant columns
            X = X.loc[:, X.nunique() > 1]

            # replace inf/nan BEFORE MFE
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

            # scale down if too large (stability + speed)
            # reduce rows (for speed)
            if X.shape[0] > 2000:
                idx = np.random.RandomState(42).choice(len(X), 2000, replace=False)
                X = X.iloc[idx]
                y = y[idx]

            # reduce features (for stability)
            if X.shape[1] > 200:
                X = X.iloc[:, :200]

            # drop problematic columns
            X = X.select_dtypes(include=[np.number])

            # remove constant columns again (post-encoding)
            X = X.loc[:, X.nunique() > 1]

            # guard tiny datasets
            if len(np.unique(y)) < 2 or X.shape[0] < 50:
                return fail_result(task_id, "degenerate dataset")


            mfe = MFE(
                groups=["statistical", "info-theory"],  # remove landmarking
                summary=["mean"]
            )

            mfe.fit(X.values, y)
            names, values = mfe.extract()

        feats = dict(zip(names, values))

        # clean outputs
        for k in feats:
            if feats[k] is None or not np.isfinite(feats[k]):
                feats[k] = 0.0

        # =========================
        # ADD TASK-LEVEL META FEATURES
        # =========================

        feats["task_type"] = "classification"

        # class imbalance (majority class ratio)
        unique, counts = np.unique(y, return_counts=True)
        feats["class_imbalance"] = np.max(counts) / len(y)

        # dimensionality (feature-to-sample ratio)
        feats["dimensionality"] = X.shape[1] / X.shape[0]

        # optional but useful
        feats["n_classes"] = len(unique)
        feats["n_features"] = X.shape[1]
        feats["n_instances"] = X.shape[0]

        feats["mfe_error"] = 0
        return feats

    except Exception as e:
        return {"mfe_error": 1}
    
def fail_result(task_id, error_msg):
    return {
        "task_id": task_id,
        "dataset_name": None,
        "model": None,
        "pipeline": None,
        "model_family": None,
        "cv_accuracy": 0.0,
        "runtime_sec": 0.0,
        "status": "failed",
        "error": error_msg,
        "mfe_error": 1
    }

def process_task(task_id):
    print("Processing: " + str(task_id))
    try:
        print(f"Starting task {task_id}")

        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute
        )

        X = pd.get_dummies(X, dummy_na=True).fillna(0).astype(float)
        X = pd.DataFrame(X)
        y = LabelEncoder().fit_transform(y)

        meta_features = extract_meta_features(X, y)
        meta_features.update(compute_landmarks(X, y))
        if len(meta_features) < 30:
            meta_features["meta_feature_padding"] = 0.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        tpot = TPOTClassifier(
            max_time_mins=TIME_LIMIT_MINS,
            random_state=RANDOM_SEED,
            cv=3,
            n_jobs=1
        )

        start = time.time()
        tpot.fit(X_train, y_train)
        runtime = time.time() - start

        model = tpot.fitted_pipeline_
        try:
            cv_score = np.mean(cross_val_score(
                model,
                X_train,
                y_train,
                cv=3,
                n_jobs=1
            ))
        except:
            cv_score = 0.0

        print(f"[DEBUG] {dataset.name} | cross_validation_score={cv_score} | runtime={runtime:.2f}s")

        final_model = model.steps[-1][1]
        model_name = type(final_model).__name__
        params = final_model.get_params()
        return {
            # identity
            "task_id": task_id,
            "dataset_name": dataset.name,

            # performance (label for meta-learning)
            "model": model_name,
            "pipeline": str(model),
            "model_family": get_model_family(model_name),
            "cv_accuracy": float(cv_score),
            "runtime_sec": runtime,

            # status
            "status": "success",

            #hyperparamers
            "hyperparameters": json.dumps({
                k: params.get(k)
                for k in ["max_depth", "n_estimators", "num_leaves"]
                if k in params
            }),
            "max_depth": params.get("max_depth"),
            "n_estimators": params.get("n_estimators"),
            "num_leaves": params.get("num_leaves"),

            # 🔥 META-FEATURES (IMPORTANT PART)
            **meta_features,
            

            # optional but VERY useful for research
            "n_rows": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(np.unique(y)),
            "majority_class_ratio": np.max(np.bincount(y)) / len(y),
            "sparsity": np.mean(X == 0),
        }

    except Exception as e:
        return fail_result(task_id, e)
        
def main():

    warnings.filterwarnings("ignore")

    suite = openml.study.get_suite("OpenML-CC18")

    task_ids = list(suite.tasks)   # already task IDs
    task_ids = task_ids[:40]
    print(task_ids)
    print("FINAL TASK COUNT:", len(task_ids))

    results = []
    start_all = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(process_task, tid): tid
            for tid in task_ids
        }

        with tqdm(total=len(task_ids), desc="Processing tasks") as pbar:

            for future in as_completed(futures):
                task_id = futures[future]

                try:
                    start_task = time.time()
                    result = future.result()
                    elapsed_task = time.time() - start_task

                    cv_val = result.get("cv_accuracy")
                    cv_str = f"{cv_val:.3f}" if isinstance(cv_val, (int, float)) else "NA"

                    pbar.set_postfix({
                        "last": result.get("dataset_name"),
                        "cv": cv_str,
                        "time": f"{elapsed_task:.1f}s"
                    })
                except Exception as e:
                    result = {
                        "task_id": task_id,
                        "status": "failed",
                        "error": str(e)
                    }
                    tqdm.write(f"FAILED | task {task_id} | {result.get('error')}")
                
                results.append(result)

                # update progress bar
                pbar.update(1)

                # optional: show live info
                if result["status"] == "success":
                    pbar.set_postfix({
                        "last": result["dataset_name"],
                        "cv": f"{result['cv_accuracy']:.3f}"
                    })
                else:
                    tqdm.write(f"FAILED | task {task_id} | {result.get('error')}")
                    pbar.set_postfix({
                        "last": f"task {task_id}",
                        "status": "failed"
                    })

                # save incrementally
                tmp_path = OUTPUT_PATH + ".tmp"
                pd.DataFrame(results).to_csv(tmp_path, index=False)
                os.replace(tmp_path, OUTPUT_PATH)


    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print("Final CSV saved. (Maybe)")  
    print("\nALL DONE")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()