import openml
import pandas as pd
import numpy as np
import time
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
from pymfe.mfe import MFE
from openml.tasks import TaskType

from concurrent.futures import ProcessPoolExecutor, as_completed

RANDOM_SEED = 42
TIME_LIMIT_MINS = 5

MAX_WORKERS = 2   # 🔥 SAFE for Mac (increase to 3–4 if M-series Pro/Max)
CACHE_DIR = "openml_cache"

def extract_meta_features(X, y):
    try:
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            for col in X.select_dtypes(include=["object", "category"]):
                X[col] = X[col].astype("category").cat.codes

        X = np.asarray(X)
        y = np.asarray(y)

        mfe = MFE(
            groups=["statistical", "info-theory"],  # fast subset
            summary=["mean"]
        )

        mfe.fit(X, y)
        names, values = mfe.extract()

        return {
            k: (0.0 if v is None or np.isnan(v) or np.isinf(v) else v)
            for k, v in dict(zip(names, values)).items()
        }

    except Exception as e:
        return {"mfe_error": 1}
    

def process_task(task_id):
    try:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute
        )

        X = pd.get_dummies(X, dummy_na=True).fillna(0)
        y = LabelEncoder().fit_transform(y)

        meta_features = extract_meta_features(X, y)

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
            n_jobs=1,              # IMPORTANT: avoid nested parallelism
            verbosity=0,
            config_dict="TPOT light"
        )

        start = time.time()
        tpot.fit(X_train, y_train)
        runtime = time.time() - start

        model = tpot.fitted_pipeline_
        test_acc = model.score(X_test, y_test)

        return {
            "task_id": task_id,
            "dataset_name": dataset.name,
            "test_accuracy": test_acc,
            "runtime_sec": runtime,
            "status": "success",
            **meta_features
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }
    
def main():

    warnings.filterwarnings("ignore")

    suite = openml.study.get_suite("OpenML-CC18")

    task_ids = [
        t for t in suite.tasks
        if openml.tasks.get_task(t).task_type == TaskType.SUPERVISED_CLASSIFICATION
    ]

    task_ids = task_ids[:40]

    results = []
    start_all = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(process_task, tid): tid for tid in task_ids}

        for i, future in enumerate(as_completed(futures), start=1):

            result = future.result()
            results.append(result)

            pd.DataFrame(results).to_csv("tpot_small_results.csv", index=False)

            elapsed = time.time() - start_all
            avg = elapsed / i
            eta = avg * (len(task_ids) - i)

            print(f"[{i}/{len(task_ids)}] done | ETA: {eta/60:.1f} min")

    print("\nALL DONE")


if __name__ == "__main__":
    main()