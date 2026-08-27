import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.load("data/X_scaled.npy")
labels = np.load("data/cluster_labels.npy")

print("Running PCA...")

pca = PCA(n_components=3, random_state=626)
X_pca = pca.fit_transform(X)

print("Explained variance:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  PC3: {pca.explained_variance_ratio_[2]:.2%}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")

try:
    feature_names = np.load(
        "data/feature_names.npy",
        allow_pickle=True
    )

    print("\nPrincipal component loadings:")

    for i, component in enumerate(pca.components_):
        print(f"\nPC{i + 1}:")

        loadings = sorted(
            zip(feature_names, component),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for feature, loading in loadings[:10]:
            print(f"  {feature:<30} {loading:+.4f}")

except FileNotFoundError:
    print("\nfeature_names.npy not found; skipping loadings.")

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="tab20",
    s=8,
    alpha=0.6
)

plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0]:.2%})"
)
plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1]:.2%})"
)
plt.title("Million Song Dataset — K-Means Clusters")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("data/clusters_pca.png", dpi=300)
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    X_pca[:, 2],
    c=labels,
    cmap="tab20",
    s=8,
    alpha=0.6
)

ax.set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"
)
ax.set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]:.2%})"
)
ax.set_zlabel(
    f"PC3 ({pca.explained_variance_ratio_[2]:.2%})"
)
ax.set_title("Million Song Dataset — K-Means Clusters")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig("data/clusters_pca_3d.png", dpi=300)
plt.show()