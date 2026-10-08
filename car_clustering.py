import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. LOAD DATASET
df = pd.read_csv("car details v4.csv")
print("\n========================================")
print("       CAR CLUSTERING PROJECT")
print("========================================")

print("\nDataset shape:", df.shape)
# 2. DATA PREPROCESSING
# Convert Engine to numeric
df["Engine"] = df["Engine"].str.replace(
    " cc", "", regex=False
)

df["Engine"] = pd.to_numeric(
    df["Engine"],
    errors="coerce"
)
# Extract numerical value from Max Power
df["Max Power"] = df["Max Power"].str.extract(
    r"(\d+\.?\d*)"
)[0]

df["Max Power"] = pd.to_numeric(
    df["Max Power"],
    errors="coerce"
)
# Select features
features = [
    "Price",
    "Kilometer",
    "Engine",
    "Max Power"
]
data = df[features].dropna()
print("\nFeatures used:")
for feature in features:
    print("-", feature)
print("\nCars after preprocessing:", len(data))

# 3. STANDARDIZATION
scaler = StandardScaler()
X = scaler.fit_transform(data)
# 4. K-MEANS CLUSTERING
print("\n========================================")
print("           K-MEANS CLUSTERING")
k_values = range(2, 11)
inertia = []
for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X)
    inertia.append(km.inertia_)
    print(
        "K =", k,
        "  Inertia =", round(km.inertia_, 2)
    )

# =====================================================
# 5. ELBOW METHOD
plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    inertia,
    marker="o"
)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.savefig( "results/elbow_method.png")
plt.show()

# 6. FINAL K-MEANS CLUSTERING
# Selected from the elbow graph
best_k = 4
print("\nSelected K =", best_k)
final_km = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
labels = final_km.fit_predict(X)
data["Cluster"] = labels
print("\nK-Means Cluster Sizes:")
for i in range(best_k):
    count = sum(labels == i)
    print(
        "Cluster", i,
        ":", count, "cars"
    )
print("\nK-Means clustering completed.")
# 7. K-MEANS CLUSTER VISUALIZATION
plt.figure(figsize=(8, 6))
plt.scatter(
    data["Price"],
    data["Max Power"],
    c=data["Cluster"],
    cmap="viridis"
)

plt.xlabel("Price")
plt.ylabel("Max Power (bhp)")
plt.title("Car Clusters using K-Means")
plt.grid()

plt.savefig(
    "results/kmeans_clusters.png"
)

plt.show()

# 8. HIERARCHICAL CLUSTERING
print("\n========================================")
print("      HIERARCHICAL CLUSTERING")
print("========================================")
# Take 100 cars for a readable dendrogram
sample = data[features].sample(
    100,
    random_state=42
)
# Standardize sample
X_sample = StandardScaler().fit_transform(
    sample
)
# Perform hierarchical clustering
linked = linkage(
    X_sample,
    method="ward"
)
print("\nMethod used: Agglomerative Clustering")
print("Linkage method: Ward")
print("Number of sample cars:", len(sample))
# Create hierarchical clusters
hierarchical_labels = fcluster(
    linked,
    t=best_k,
    criterion="maxclust"
)
print("Number of clusters:", best_k)
print("\nHierarchical Cluster Sizes:")
for i in range(1, best_k + 1):

    count = sum(
        hierarchical_labels == i
    )
    print(
        "Cluster", i,
        ":", count, "cars"
    )
print("\nHierarchical clustering completed.")
# =====================================================
# 9. DENDROGRAM
# =====================================================
plt.figure(figsize=(10, 5))
dendrogram(
    linked,
    truncate_mode="lastp",
    p=25
)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")
plt.grid()
plt.savefig( "results/dendrogram.png")
plt.show()
print("\nDendrogram completed.")