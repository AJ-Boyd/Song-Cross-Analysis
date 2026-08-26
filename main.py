"""
9/27/25
AJ Boyd
"""
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# global constants
DATA_FILE = "data/msd.csv"
N_CLUSTERS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 626

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

def weighted_mean(arr, weights):
    """Compute weighted mean of an array with given weights"""
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

# test/train split
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

df["split"] = "train"
df.loc[test_df.index, "split"] = "test"

# save split data for reference
df.to_csv("data/msd_split.csv", index=False)

print(f"Total songs: {len(df)}")
print(f"Training songs: {len(train_df)}")
print(f"Test songs: {len(test_df)}")

# separate scalar and array features
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

if __name__ == "__main__":
    # process array features to compute weighted means and stds
    weighted_arrays = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing arrays"):
        song_features = []

        for feat in array_features:
            vals = json_to_array(row[f"{feat}_start"])
            confs = json_to_array(row[f"{feat}_confidence"])

            w_mean = weighted_mean(vals, confs)

            if len(vals) > 0 and len(confs) > 0 and len(vals) == len(confs) and np.sum(confs) > 0:
                w_var = weighted_mean((vals - w_mean) ** 2, confs)
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
        loudness_mean = np.mean(loudness_max) if len(loudness_max) > 0 else 0.0

        # combine
        segment_features.append(
            np.concatenate([timbre_mean, pitches_mean, [loudness_mean]])
        )

    segment_features = np.array(segment_features)

    # ----- Combine all features -----
    print("Combining features...")
    X = np.hstack([scalars, weighted_arrays, segment_features])
    print(f"Feature matrix shape: {X.shape}")

    # ----- Check for NaNs / Infs -----
    if not np.isfinite(X).all():
        print("\n===== INVALID VALUES DETECTED =====")

        rows, cols = np.where(~np.isfinite(X))
        print(f"{len(np.unique(rows))} songs contain NaNs or infinite values.\n")

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

        raise RuntimeError("Invalid values detected in feature matrix.")

    train_indices = train_df.index.to_numpy()
    test_indices = test_df.index.to_numpy()

    X_train = X[train_indices]
    X_test = X[test_indices]

    # ----- Normalize -----
    print("Normalizing features...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----- Clustering -----
    print(f"Clustering training data into {N_CLUSTERS} clusters...")

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    train_labels = kmeans.fit_predict(X_train_scaled)
    test_labels = kmeans.predict(X_test_scaled)

    labels = np.empty(len(df), dtype=int)
    labels[train_indices] = train_labels
    labels[test_indices] = test_labels

    df["cluster_label"] = labels

    # preserve original song ordering
    X_scaled = np.empty_like(X)
    X_scaled[train_indices] = X_train_scaled
    X_scaled[test_indices] = X_test_scaled

    # ----- Save clustering data for analysis/visualization -----
    np.save("data/X_scaled.npy", X_scaled)
    np.save("data/cluster_labels.npy", labels)
    np.save("data/cluster_centers.npy", kmeans.cluster_centers_)

    print("\nCluster distribution:")
    print(df["cluster_label"].value_counts().sort_index())

    # ----- Save clustered data -----
    df.to_csv("data/msd_clustered.csv", index=False)

    print("✅ Saved clustered data to data/msd_clustered.csv")