import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_model_name_from_string(pipeline_str):
    """
    If your CSV only has string pipelines, extract the final model name.
    Example: "...RandomForestClassifier(...)" -> "RandomForestClassifier"
    """
    try:
        # crude but effective extraction
        return pipeline_str.split('(')[0].split()[-1]
    except:
        return "Unknown"


def main():
    # === Load your CSV ===
    df = pd.read_csv("tpot_small_results.csv")

    print("\nLoaded data shape:", df.shape)
    print(df.columns.tolist())

    # Count algorithms
    counts = df["model"].value_counts()

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=counts.index,
        y=counts.values,
        color="steelblue"
    )

    plt.title("Algorithm Selection Frequency (Before Grouping)")
    plt.xlabel("Algorithm")
    plt.ylabel("Number of Datasets")
    plt.xticks(rotation=45, ha='right')

    # annotate
    for i, v in enumerate(counts.values):
        plt.text(i, v + 0.2, str(v), ha='center')

    plt.tight_layout()
    plt.savefig("algo_distribution_before.png")
    plt.show()

    # Group rare classes
    threshold = 3
    rare_classes = counts[counts < threshold].index

    df["model_grouped"] = df["model"].apply(
        lambda x: x if x not in rare_classes else "Other"
    )

    grouped_counts = df["model_grouped"].value_counts()

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=grouped_counts.index,
        y=grouped_counts.values,
        color="darkorange"
    )

    plt.title("Algorithm Selection Frequency (After Grouping Rare Classes)")
    plt.xlabel("Algorithm")
    plt.ylabel("Number of Datasets")
    plt.xticks(rotation=45, ha='right')

    for i, v in enumerate(grouped_counts.values):
        plt.text(i, v + 0.2, str(v), ha='center')

    plt.tight_layout()
    plt.savefig("algo_distribution_after.png")
    plt.show()

    plt.figure(figsize=(8, 6))

    print("\nAvailable columns:")
    print(df.columns.tolist())

    plt.figure(figsize=(10, 5))

    counts = df["model_grouped"].value_counts()

    sns.barplot(
        x=counts.index,
        y=counts.values,
        color="steelblue"
    )

    plt.title("Distribution of AutoML-Selected Algorithms")
    plt.xlabel("Algorithm")
    plt.ylabel("Number of Datasets")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    # choose correct accuracy column
    target = "test_accuracy"

    features = ["class_ent", "ns_ratio", "gravity", "eq_num_attr"]

    for f in features:
        if f in df.columns:
            plt.figure(figsize=(6,4))
            sns.scatterplot(data=df, x=f, y=target, hue="model_grouped")
            plt.title(f"{f} vs Accuracy")
            plt.tight_layout()
            plt.show()

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="mut_inf.mean",
        y="test_accuracy",
        hue="model_grouped",
        palette="tab10"
    )

    plt.title("Mutual Information vs Accuracy")
    plt.xlabel("Mutual Information (signal strength)")
    plt.ylabel("Test Accuracy")

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="one_nn.mean",
        y="test_accuracy",
        hue="model_grouped",
        palette="tab10"
    )

    plt.title("1-NN Performance vs AutoML Accuracy")
    plt.xlabel("1-NN Accuracy (dataset easiness)")
    plt.ylabel("AutoML Test Accuracy")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))

    numeric_df = df.select_dtypes(include=["float64", "int64"])

    corr = numeric_df.corr()

    sns.heatmap(corr, cmap="coolwarm", center=0)

    plt.title("Meta-Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

    target_corr = numeric_df.corr()["test_accuracy"].drop("test_accuracy")

    top = target_corr.reindex(target_corr.abs().sort_values(ascending=False).head(10).index)

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=top.values,
        y=top.index,
        color="darkgreen"
    )

    plt.title("Top Meta-Features Influencing Accuracy")
    plt.xlabel("Correlation with Accuracy")
    plt.ylabel("Meta-Feature")

    plt.tight_layout()
    plt.show()

    features = df.select_dtypes(include=["float64", "int64"]).drop(columns=["test_accuracy"])

    X_scaled = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="model_grouped",
        palette="tab10"
    )

    plt.title("PCA of Meta-Features (Dataset Similarity Space)")
    plt.tight_layout()
    plt.show()
    # === Compute naive baseline ===
    most_common = grouped_counts.idxmax()
    frequency = grouped_counts.max() / len(meta_dataset)

    print("\n=== Naive Baseline ===")
    print(f"Most common algorithm: {most_common}")
    print(f"Frequency: {frequency:.4f}")

    # === Save grouped dataset ===
    meta_dataset.to_csv("meta_dataset_grouped.csv", index=False)
    print("\nSaved meta_dataset_grouped.csv")

    plt.figure(figsize=(12, 6))


if __name__ == "__main__":
    main()