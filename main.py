"""
main.py
-------
Task 02 : Customer Segmentation using K-Means Clustering
Author  : <Your Name>
Dataset : Mall_Customers.csv

This script performs the full pipeline:
    1. Load and inspect the data
    2. Clean and explore the data (EDA)
    3. Visualize relationships between features
    4. Find the optimal number of clusters (Elbow Method)
    5. Train a K-Means model
    6. Visualize the resulting customer segments
    7. Save all plots to the /images folder for the README / portfolio

Run with:
    python main.py
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
# pandas    -> loading and manipulating tabular data (the CSV file)
# numpy     -> fast numerical operations on arrays
# matplotlib.pyplot -> creating charts and graphs
# seaborn   -> nicer, higher-level statistical charts (built on matplotlib)
# sklearn.cluster.KMeans -> the clustering algorithm itself
# pathlib.Path -> safe, OS-independent handling of file paths
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path

# Make the plots look clean and consistent everywhere in the script
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# ============================================================
# 2. DEFINE FILE PATHS (so the script works on any computer)
# ============================================================
# Path(__file__).resolve().parent -> the folder that main.py lives in.
# We build the data/ and images/ paths relative to that, instead of
# hard-coding "C:/Users/you/..." which would break on other machines.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)  # create images/ if it doesn't exist yet


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the Mall Customers dataset from a CSV file.

    Parameters
    ----------
    path : Path
        Location of the CSV file on disk.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Why a function and a try/except?
    ---------------------------------
    Wrapping file I/O in a function makes it reusable and testable.
    The try/except gives a clear, beginner-friendly error message
    instead of a confusing Python traceback if the file is missing.
    """
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}\n")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find the dataset at: {path}\n"
            "Make sure Mall_Customers.csv is inside the 'data' folder."
        )


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic exploratory information about the dataset:
    first rows, shape, column info, summary statistics,
    missing values, and duplicate rows.
    """
    print("=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head(), "\n")

    print("=" * 60)
    print("DATASET SHAPE (rows, columns)")
    print("=" * 60)
    print(df.shape, "\n")

    print("=" * 60)
    print("COLUMN INFO (data types, non-null counts)")
    print("=" * 60)
    df.info()
    print()

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe(), "\n")

    print("=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    print(df.isnull().sum(), "\n")

    print("=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    print(f"Number of duplicate rows: {df.duplicated().sum()}\n")


def visualize_eda(df: pd.DataFrame) -> None:
    """
    Create exploratory visualizations and save them to the images folder:
        - Age distribution (histogram)
        - Gender count (countplot)
        - Annual Income vs Spending Score (scatter plot)
        - Pairplot of numeric features
        - Correlation heatmap
    """
    # --- Histogram: Age distribution ---
    plt.figure()
    sns.histplot(df["Age"], bins=20, kde=True, color="steelblue")
    plt.title("Distribution of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "age_distribution.png", dpi=150)
    plt.close()

    # --- Countplot: Gender ---
    plt.figure()
    sns.countplot(x="Gender", data=df, palette="Set2")
    plt.title("Customer Count by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "gender_count.png", dpi=150)
    plt.close()

    # --- Scatter plot: Income vs Spending Score ---
    plt.figure()
    sns.scatterplot(
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Gender",
        data=df,
        palette="deep",
        s=60,
    )
    plt.title("Annual Income vs Spending Score")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend(title="Gender")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "income_vs_spending.png", dpi=150)
    plt.close()

    # --- Pairplot of numeric features ---
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    pairplot_fig = sns.pairplot(df[numeric_cols])
    pairplot_fig.fig.suptitle("Pairwise Relationships", y=1.02)
    pairplot_fig.savefig(IMAGES_DIR / "pairplot.png", dpi=150)
    plt.close("all")

    # --- Correlation heatmap ---
    plt.figure()
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"EDA plots saved to: {IMAGES_DIR}\n")


def find_optimal_k(X: np.ndarray, max_k: int = 10) -> None:
    """
    Use the Elbow Method to help choose the number of clusters (k).

    For each k from 1 to max_k, we fit a KMeans model and record its
    'inertia_' (a.k.a. WCSS - Within-Cluster Sum of Squares), which
    measures how tightly the points are grouped around their centroid.
    Inertia always decreases as k increases; we look for the "elbow"
    point where adding more clusters stops giving a big improvement.
    """
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, max_k + 1), wcss, marker="o", linestyle="--", color="darkorange")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Inertia)")
    plt.xticks(range(1, max_k + 1))
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "elbow_method.png", dpi=150)
    plt.close()

    print(f"Elbow method plot saved to: {IMAGES_DIR / 'elbow_method.png'}\n")


def train_kmeans(X: np.ndarray, n_clusters: int) -> KMeans:
    """
    Train a K-Means model with the chosen number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (Annual Income, Spending Score).
    n_clusters : int
        Number of clusters chosen from the elbow plot.

    Returns
    -------
    KMeans
        The fitted model, which also stores cluster centers and labels.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",   # smarter initial centroid placement than random
        random_state=42,    # ensures the same result every run (reproducibility)
        n_init=10,          # number of times the algorithm runs with different seeds
    )
    kmeans.fit(X)
    return kmeans


def visualize_clusters(df: pd.DataFrame, X: np.ndarray, kmeans: KMeans) -> None:
    """
    Plot the customer segments produced by K-Means, along with their
    centroids, using distinct colors, labels, and a legend.
    """
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("Set1", n_colors=len(np.unique(labels)))

    for cluster_id in np.unique(labels):
        plt.scatter(
            X[labels == cluster_id, 0],
            X[labels == cluster_id, 1],
            s=60,
            color=palette[cluster_id],
            label=f"Cluster {cluster_id + 1}",
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=250,
        c="black",
        marker="X",
        label="Centroids",
    )

    plt.title("Customer Segments (K-Means Clustering)")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "customer_clusters.png", dpi=150)
    plt.close()

    print(f"Cluster plot saved to: {IMAGES_DIR / 'customer_clusters.png'}\n")


def main() -> None:
    """Run the full customer segmentation pipeline end to end."""
    df = load_data(DATA_PATH)
    explore_data(df)
    visualize_eda(df)

    # Select the two features used for segmentation.
    # .values converts the pandas columns into a NumPy array, which
    # scikit-learn's KMeans expects as input.
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    find_optimal_k(X, max_k=10)

    # Based on the elbow plot for this dataset, 5 clusters is a strong choice.
    optimal_k = 5
    kmeans = train_kmeans(X, optimal_k)

    # Attach the cluster label to each customer for business analysis.
    df["Cluster"] = kmeans.labels_

    visualize_clusters(df, X, kmeans)

    print("Cluster sizes:")
    print(df["Cluster"].value_counts().sort_index(), "\n")

    print("Pipeline finished successfully. Check the 'images' folder for all charts.")


if __name__ == "__main__":
    main()
