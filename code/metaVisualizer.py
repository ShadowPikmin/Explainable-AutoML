import pandas as pd 
import matplotlib.pyplot as plt
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
PNG_DIR = ROOT / "png"

os.makedirs(PNG_DIR, exist_ok=True)

df = pd.read_csv(DATA_DIR / "tpot_results.csv")
label_col = "model_family"

counts = df[label_col].value_counts()
proportions = df[label_col].value_counts(normalize=True)

most_common_algo = counts.idxmax()
baseline_accuracy = proportions.max()

print("Most common algorithm:", most_common_algo)
print("Naive baseline accuracy:", round(baseline_accuracy, 3))

plt.figure()
proportions.plot(kind="bar")

plt.title("Algorithm Distribution (Proportion)")
plt.xlabel("Algorithm")
plt.ylabel("Proportion")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(PNG_DIR, "algorithm_distribution.png"), dpi=300)
plt.show()
plt.close()

plt.figure()
proportions.plot(kind="bar")

plt.axhline(y=baseline_accuracy, linestyle="--")
plt.title(f"Naive Baseline = {baseline_accuracy:.2f} ({most_common_algo})")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(PNG_DIR, "baseline_distribution.png"), dpi=300)
plt.show()
plt.close()

print(f"The most frequently selected algorithm is {most_common_algo} "
    + f" appearing in {baseline_accuracy * 100:.2f}%. This defines " 
    + f" defines a naive baseline accuracy of {baseline_accuracy:.2f}")