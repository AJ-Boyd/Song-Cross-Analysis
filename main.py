"""
9/27/25
AJ Boyd
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm

DATA_FILE = "data/msd.csv"
N_CLUSTERS = 50  # adjust as needed

# helper functions
def json_to_array(s):
    """Convert JSON string to numpy array"""
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return np.array(arr)
        return np.array([])
    except:
        return np.array([])
    
def weighted_mean(arr, weights, name="unknown"):
    if len(arr) == 0:
        # print(f"{name}: Empty values")
        return 0.0

    if len(weights) == 0:
        # print(f"{name}: Empty weights")
        return 0.0

    if len(arr) != len(weights):
        # print(f"{name}: Length mismatch {len(arr)} vs {len(weights)}")
        return 0.0

    if np.sum(weights) == 0:
        # print(f"{name}: Zero weights")
        return 0.0

    return np.average(arr, weights=weights)

# pre-requisite: preprocess_data.py should have been run to generate data/msd.csv
# Load and define data
df = pd.read_csv(DATA_FILE)
scalar_features = [
    "duration", "loudness", "tempo",
    "key", "key_confidence", "mode", "mode_confidence",
    "time_signature", "time_signature_confidence",
    "song_hotttnesss", "start_of_fade_out", "end_of_fade_in"
]
df[scalar_features] = df[scalar_features].fillna(df[scalar_features].mean())
scalars = df[scalar_features].fillna(0).values
array_features = [
    "bars", "beats", "tatums", "sections", "segments"
]
weighted_arrays = []

if __name__ == "__main__":
    # process array feautures to compute weighted means and stds
    weighted_arrays = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing arrays"):
        song_features = []

        for feat in array_features:
            vals = json_to_array(row[f"{feat}_start"])
            confs = json_to_array(row[f"{feat}_confidence"])

            w_mean = weighted_mean(vals, confs, feat)

            if len(vals) > 0 and len(confs) > 0 and np.sum(confs) > 0:
                w_var = weighted_mean((vals - w_mean) ** 2, confs, feat)
                w_std = np.sqrt(w_var)
            else:
                w_std = 0.0

            song_features.extend([w_mean, w_std])

        weighted_arrays.append(song_features)

    weighted_arrays = np.array(weighted_arrays)

    # ----- Segment-level features (timbre, pitches, loudness) -----
    segment_features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing segments"):
        # segments_timbre
        timbre = json_to_array(row["segments_timbre"])
        if timbre.ndim == 2 and timbre.shape[0] > 0:
            timbre_mean = timbre.mean(axis=0)
        else:
            timbre_mean = np.zeros(12)
        # segments_pitches
        pitches = json_to_array(row["segments_pitches"])
        if pitches.ndim == 2 and pitches.shape[0] > 0:
            pitches_mean = pitches.mean(axis=0)
        else:
            pitches_mean = np.zeros(12)
        # segments_loudness_max
        loudness_max = json_to_array(row["segments_loudness_max"])
        loudness_mean = np.mean(loudness_max) if len(loudness_max) > 0 else 0
        # combine
        segment_features.append(np.concatenate([timbre_mean, pitches_mean, [loudness_mean]]))
    segment_features = np.array(segment_features)

    # ----- Combine all features -----
    print("Combining features...")
    X = np.hstack([scalars, weighted_arrays, segment_features])
    print(f"Feature matrix shape: {X.shape}")

# ----- Check for NaNs / Infs -----
if np.isnan(X).any():
    print("\n===== NaN DETECTED =====")

    rows, cols = np.where(np.isnan(X))

    print(f"{len(np.unique(rows))} songs contain NaNs.\n")

    for row in np.unique(rows)[:10]:
        print("=" * 60)
        print(f"Row: {row}")
        print(f"Title : {df.loc[row, 'title']}")
        print(f"Artist: {df.loc[row, 'artist_name']}")

        print("\nBars")
        print(json_to_array(df.loc[row, "bars_start"]))
        print(json_to_array(df.loc[row, "bars_confidence"]))

        print("\nBeats")
        print(json_to_array(df.loc[row, "beats_start"]))
        print(json_to_array(df.loc[row, "beats_confidence"]))

        print("\nTatums")
        print(json_to_array(df.loc[row, "tatums_start"]))
        print(json_to_array(df.loc[row, "tatums_confidence"]))

        print("\nSections")
        print(json_to_array(df.loc[row, "sections_start"]))
        print(json_to_array(df.loc[row, "sections_confidence"]))

        print("\nSegments")
        print(json_to_array(df.loc[row, "segments_start"]))
        print(json_to_array(df.loc[row, "segments_confidence"]))

# ----- Normalize -----
print("Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----- Clustering -----
print(f"Clustering into {N_CLUSTERS} clusters...")
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

# Save clustering data for analysis/visualization
np.save("data/X_scaled.npy", X_scaled)
np.save("data/cluster_labels.npy", labels)
np.save("data/cluster_centers.npy", kmeans.cluster_centers_)

df["cluster_label"] = labels

print("Cluster distribution:")
print(df["cluster_label"].value_counts().sort_index())

# ----- Save clustered data -----
df.to_csv("data/msd_clustered.csv", index=False)

print("✅ Saved clustered data to data/msd_clustered.csv")
    # # ran into NaN issues, so let's check
    # if np.isnan(X).any():
    #     print("\n===== NaN DETECTED =====")

    #     # ----- Debug NaNs -----
    # rows, cols = np.where(np.isnan(X))

    # if len(rows) > 0:
    #     print(f"{len(np.unique(rows))} songs contain NaNs.\n")

    #     for row in np.unique(rows)[:10]:      # print first 10 bad songs
    #         print("=" * 60)
    #         print(f"Row: {row}")
    #         print(f"Title : {df.loc[row, 'title']}")
    #         print(f"Artist: {df.loc[row, 'artist_name']}")

    #         print("\nBars")
    #         print(json_to_array(df.loc[row, "bars_start"]))
    #         print(json_to_array(df.loc[row, "bars_confidence"]))

    #         print("\nBeats")
    #         print(json_to_array(df.loc[row, "beats_start"]))
    #         print(json_to_array(df.loc[row, "beats_confidence"]))

    #         print("\nTatums")
    #         print(json_to_array(df.loc[row, "tatums_start"]))
    #         print(json_to_array(df.loc[row, "tatums_confidence"]))

    #         print("\nSections")
    #         print(json_to_array(df.loc[row, "sections_start"]))
    #         print(json_to_array(df.loc[row, "sections_confidence"]))

    #         print("\nSegments")
    #         print(json_to_array(df.loc[row, "segments_start"]))
    #         print(json_to_array(df.loc[row, "segments_confidence"]))

    #     # raise RuntimeError("Stopping for debugging.")
        
    #     # ----- Normalize -----
    #     print("Normalizing features...")
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)

    #     # ----- Clustering -----
    #     print(f"Clustering into {N_CLUSTERS} clusters...")
    #     kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    #     labels = kmeans.fit_predict(X_scaled)

    #     df["cluster_label"] = labels
    #     print("Cluster distribution:\n", df["cluster_label"].value_counts())

    #     # ----- Save clustered data -----
    #     df.to_csv("data/msd_clustered.csv", index=False)
    #     print("✅ Saved clustered data to data/msd_clustered.csv")