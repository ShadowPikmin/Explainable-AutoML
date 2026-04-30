import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

os.makedirs(PNG_DIR, exist_ok=True)

df = pd.read_csv(DATA_DIR / "tpot_results_adjusted.csv")
label_col = "algorithm" 

print(df[["task_id","algorithm"]] )

algorithm, counts = np.unique(df["algorithm"], return_counts=True)

fig, ax = plt.subplots()

font = {'size': 10}

ax.bar(algorithm, counts)
ax.set_title("Best Algorithm occurence")
ax.set_ylabel("Frequency")
ax.set_xlabel("Algorihtm")

ax.tick_params(axis='x', which='both', labelsize=5)

plt.show()


proportion = df["algorithm"].value_counts(normalize=True)

fig, ax = plt.subplots()

font = {'size': 10}

ax.bar(algorithm, proportion)
ax.set_title("Best Algorithm occurence (proportion)")
ax.set_ylabel("Proportion")
ax.set_xlabel("Algorihtm")

ax.tick_params(axis='x', which='both', labelsize=5)

plt.show()

print("best algorithm: " + str(df["algorithm"].value_counts().idxmax()))
print("Baseline accuracy: " + str(proportion.max()))