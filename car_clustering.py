import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. LOAD DATASET

df = pd.read_csv("car details v4.csv")
print("\nDataset shape:", df.shape)

# 2. DATA PREPROCESSING

# Convert Engine to numeric
df["Engine"] = df["Engine"].str.replace(" cc", "", regex=False)
df["Engine"] = pd.to_numeric(df["Engine"], errors="coerce")

# Extract numerical value from Max Power
df["Max Power"] = df["Max Power"].str.extract(
    r"(\d+\.?\d*)"
)[0]

df["Max Power"] = pd.to_numeric(
    df["Max Power"],
    errors="coerce"
)

# Select important features
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
plt.savefig("results/elbow_method.png")
plt.show()
# =====================================================
# 6. FINAL K-MEANS
# Choose K from the elbow graph
best_k = 4

print("\nSelected K =", best_k)

final_km = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

labels = final_km.fit_predict(X)
data["Cluster"] = labels
print("\nCluster sizes:")

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

plt.savefig("results/kmeans_clusters.png")
plt.show()

# 8. HIERARCHICAL CLUSTERING


print("\n========================================")
print("      HIERARCHICAL CLUSTERING")
# Take 100 cars for a readable dendrogram
sample = data[features].sample(
    100,
    random_state=42
)
X_sample = StandardScaler().fit_transform(sample)
linked = linkage(
    X_sample,
    method="ward"
)

print("\nHierarchical clustering completed.")
print("\n========================================")
print("               DENDROGRAM")

print("Dendrogram generated using 100 sample cars.")
# 9. DENDROGRAM
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
plt.savefig("results/dendrogram.png")
plt.show()
