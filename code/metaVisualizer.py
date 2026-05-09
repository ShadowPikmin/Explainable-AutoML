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

#Adds value labels to bar plot
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])

df_tpot = pd.read_csv(DATA_DIR / "tpot_results.csv")
df_metafeatures = pd.read_csv(DATA_DIR / "metafeatures.csv")

#Frequency report
algorithm, counts = np.unique(df_tpot['algorithm'], return_counts = True)

plt.barh(algorithm, counts)
plt.title('Best Algorithms (Frequency)')

plt.ylabel('Algorithms')
plt.yticks(fontsize=8)

plt.xlabel('Frequency')
plt.savefig(PNG_DIR / "AlgorithmFrequency.png", dpi=300, bbox_inches="tight")
plt.show()

#Proportion report 
# Combine both series into one DataFrame
proportion_report = df_tpot['algorithm'].value_counts(normalize = True)

algo = proportion_report.index
proportions = proportion_report.values

baseline = max(proportions)

plt.barh(algo, proportions)

plt.title('Best Algorithms (Proportion)')

plt.axvline(x=baseline, color='red', linestyle='--', linewidth=2, label=f'baseline: {baseline}')

plt.ylabel('Algorithms')
plt.yticks(fontsize=8)

plt.xlabel('Proportion')
plt.legend()
plt.savefig(PNG_DIR / "AlgorithmProportion.png", dpi=300, bbox_inches="tight")
plt.show()

#Accuracy report 
task = df_tpot['task'].values
accuracy = df_tpot['cv_accuracy'].values

plt.bar(task, accuracy)

plt.title("Final Accuracy on Tasks")
plt.xlabel("Tasks")
plt.xticks(rotation='vertical', fontsize = 4)

plt.ylabel("CV Accuracy")
plt.savefig(PNG_DIR / "TPOT_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

#Runtime report 
runtime = df_tpot['runtime'].values

plt.bar(task, runtime)

plt.title("Runtime on each Task")
plt.xlabel("Tasks")
plt.xticks(rotation='vertical', fontsize = 4)

plt.ylabel("Runtime (seconds)")
plt.savefig(PNG_DIR / "TPOT_runtime.png", dpi=300, bbox_inches="tight")
plt.show()

#Missing Data report 
# 1. Count zeros in every column
zero_counts = (df_metafeatures == 0).sum()

# 2. Filter to show only columns that have at least one zero
columns_with_zeros = zero_counts[zero_counts > 0]

print(columns_with_zeros)


#Family Data report 
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

df_family = df_tpot['algorithm'].map(family_map)
print(df_family)
# Combine both series into one DataFrame
proportion_report = df_family.value_counts(normalize = True)

algo = proportion_report.index
proportions = proportion_report.values

baseline = max(proportions)

plt.barh(algo, proportions)

plt.title('Best Algorithms (Proportion)')

plt.axvline(x=baseline, color='red', linestyle='--', linewidth=2, label=f'baseline: {baseline}')

plt.ylabel('Algorithms')
plt.yticks(fontsize=8)

plt.xlabel('Proportion')
plt.legend()
plt.savefig(PNG_DIR / "AlgorithmFamilyProportion.png", dpi=300, bbox_inches="tight")
plt.show()
