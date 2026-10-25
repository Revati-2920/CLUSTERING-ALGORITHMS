"""Practical 5: Clustering on a real-world dataset
Dataset: Wine (chemical analysis of 178 wines, 13 features) - built into scikit-learn.
Algorithms: K-Means, Agglomerative (Ward) hierarchical clustering, DBSCAN, Gaussian Mixture."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Load and scale the data (labels are NOT used for clustering)
data = load_wine()
X = StandardScaler().fit_transform(data.data)
y_true = data.target                      # used only for an external check at the end
X2 = PCA(n_components=2, random_state=42).fit_transform(X)   # for visualisation only

# 2. Choose the number of clusters (elbow + silhouette)
ks = list(range(2, 9))
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
best_k = ks[int(np.argmax(sil))]
print("k vs silhouette:", {k: round(s, 4) for k, s in zip(ks, sil)}, "-> best k =", best_k)

# 3. Fit the four algorithms
K = 3
labels = {
    "K-Means": KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(X),
    "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X),
    "DBSCAN": DBSCAN(eps=2.3, min_samples=5).fit_predict(X),
    "Gaussian Mixture": GaussianMixture(n_components=K, random_state=42).fit_predict(X),
}

# 4. Evaluate (internal metrics + adjusted Rand index as an external check)
results = {}
for name, lab in labels.items():
    mask = lab != -1                       # DBSCAN noise points are excluded from the scores
    n_clusters = len(set(lab[mask]))
    results[name] = {
        "clusters": n_clusters,
        "noise": int((~mask).sum()),
        "silhouette": silhouette_score(X[mask], lab[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], lab[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], lab[mask]),
        "ari": adjusted_rand_score(y_true, lab),
        "sizes": [int((lab == c).sum()) for c in sorted(set(lab))],
    }
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[name].items()})

# Cluster profile of the K-Means groups (mean of original, unscaled features)
import pandas as pd
df = pd.DataFrame(data.data, columns=data.feature_names)
df["cluster"] = labels["K-Means"]
key = ["alcohol", "flavanoids", "color_intensity", "proline"]
profile = df.groupby("cluster")[key].mean().round(2)
profile["size"] = df.groupby("cluster").size()
print(profile)

json.dump({"profile": profile.reset_index().values.tolist(), "results": results, "ks": ks, "inertia": inertia, "sil": sil, "best_k": best_k,
           "n": len(X), "f": X.shape[1]}, open("results.json", "w"), indent=1)

# 5. Plots
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "s-", color="tab:green"); ax[1].set_title("Silhouette score vs k"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("elbow_silhouette.png", dpi=160); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
for a, (name, lab) in zip(axes, labels.items()):
    a.scatter(X2[:, 0], X2[:, 1], c=lab, cmap="viridis", s=18)
    a.set_title(f"{name}\nSilhouette = {results[name]['silhouette']:.3f}", fontsize=10)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); plt.savefig("cluster_plots.png", dpi=160); plt.close()

plt.figure(figsize=(8, 3.6))
dendrogram(linkage(X, "ward"), truncate_mode="lastp", p=20, no_labels=True)
plt.title("Dendrogram (Ward linkage, last 20 merges)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig("dendrogram.png", dpi=160); plt.close()

names = list(results)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(names, [results[n]["silhouette"] for n in names], color="tab:blue")
ax[0].set_title("Silhouette score (higher is better)")
ax[1].bar(names, [results[n]["davies_bouldin"] for n in names], color="tab:orange")
ax[1].set_title("Davies-Bouldin index (lower is better)")
for a in ax: a.tick_params(axis="x", labelrotation=20); a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("metric_comparison.png", dpi=160); plt.close()
