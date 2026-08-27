import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

DATA_FILE = "data/msd_clustered.csv"
X_FILE = "data/X_scaled.npy"
CENTERS_FILE = "data/cluster_centers.npy"

N_REPRESENTATIVES = 5

df = pd.read_csv(DATA_FILE)
X = np.load(X_FILE)
centers = np.load(CENTERS_FILE)

print("===== CLUSTER PROFILES =====")

for cluster in sorted(df["cluster_label"].unique()):

    indices = np.where(df["cluster_label"].values == cluster)[0]
    cluster_X = X[indices]
    center = centers[cluster]

    # Distance from every song in cluster to its centroid
    distances = np.linalg.norm(cluster_X - center, axis=1)

    # Closest songs = most representative
    closest = np.argsort(distances)[:N_REPRESENTATIVES]

    print("\n" + "=" * 70)
    print(f"CLUSTER {cluster}")
    print(f"Songs: {len(indices)}")

    print("\nRepresentative songs:")

    for rank, position in enumerate(closest, 1):
        row = df.iloc[indices[position]]

        title = row["title"]
        artist = row["artist_name"]
        year = row["year"]

        if pd.isna(title) or not str(title).strip():
            title = "[Unknown title]"

        if pd.isna(artist) or not str(artist).strip():
            artist = "[Unknown artist]"

        print(
            f"  {rank}. {title} — {artist} ({year})"
            f" | distance: {distances[position]:.3f}"
        )