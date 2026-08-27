# preprocess_data_final.py

import os
import glob
import h5py
import pandas as pd
import json
from tqdm import tqdm
import numpy as np

DATA_DIR = "data/MillionSongSubset"  # adjust to your folder
OUTPUT_FILE = "data/msd.csv"

def array_to_json(arr):
    if arr is None or len(arr) == 0:
        return "[]"
    return json.dumps(arr.tolist())

def extract_fields_final(h5_path):
    row = {}

    with h5py.File(h5_path, 'r') as h5:
        #----- Track ID -----
        songs = h5["analysis/songs"][0]
        row["track_id"] = songs["track_id"].decode("utf-8", errors="ignore")

        # ----- Metadata -----
        songs = h5["metadata/songs"][0]
        musicbrainz = h5["musicbrainz/songs"][0]

        metadata_fields = [
            ("title", songs["title"]),
            ("artist_name", songs["artist_name"]),
            ("release", songs["release"]),
            ("year", musicbrainz["year"])
        ]

        for col, val in metadata_fields:
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="ignore")
            row[col] = val

        # ----- Artist terms only -----
        artist_terms_fields = [
            ("artist_terms", "metadata/artist_terms"),
            ("artist_terms_freq", "metadata/artist_terms_freq"),
            ("artist_terms_weight", "metadata/artist_terms_weight")
        ]
        for col, path in artist_terms_fields:
            if path in h5:
                arr = h5[path][()]
                if arr.dtype.kind == 'S':
                    arr = [v.decode("utf-8") for v in arr]
                row[col] = json.dumps(arr.tolist()) if isinstance(arr, np.ndarray) else json.dumps(arr)
            else:
                row[col] = "[]"

        # ----- Scalar acoustic features -----

        scalar_fields = [
            "duration",
            "danceability",
            "energy",
            "loudness",
            "tempo",
            "key",
            "key_confidence",
            "mode",
            "mode_confidence",
            "time_signature",
            "time_signature_confidence",
            "start_of_fade_out",
            "end_of_fade_in"
        ]
        songs = h5["analysis/songs"][0]
        row["track_id"] = songs["track_id"].decode("utf-8", errors="ignore")
        for field in scalar_fields:
            row[field] = float(songs[field])

        row["song_hotttnesss"] = float(h5["metadata/songs"][0]["song_hotttnesss"])

        # ----- Structural arrays (bars, beats, tatums, sections, segments) -----
        structural_features = ["bars", "beats", "tatums", "sections", "segments"]
        for feat in structural_features:
            start_path = f"analysis/{feat}_start"
            conf_path  = f"analysis/{feat}_confidence"
            row[f"{feat}_start"] = array_to_json(h5[start_path][()]) if start_path in h5 else "[]"
            row[f"{feat}_confidence"] = array_to_json(h5[conf_path][()]) if conf_path in h5 else "[]"

        # ----- Segment-specific features -----
        segment_arrays = [
            ("segments_timbre", "analysis/segments_timbre"),
            ("segments_pitches", "analysis/segments_pitches"),
            ("segments_loudness_start", "analysis/segments_loudness_start"),
            ("segments_loudness_max", "analysis/segments_loudness_max"),
            ("segments_loudness_max_time", "analysis/segments_loudness_max_time")
        ]
        for col, path in segment_arrays:
            if path in h5:
                row[col] = array_to_json(h5[path][()])
            else:
                row[col] = "[]"

    return row


if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True)
    print(f"Found {len(all_files)} HDF5 files.")

    rows = []
    for f in tqdm(all_files, desc="Processing HDF5 files"):
        try:
            rows.append(extract_fields_final(f))
        except Exception as e:
            print(f"Error in {f}: {e}")

    df = pd.DataFrame(rows)

    print("\n===== METADATA CHECK =====")
    print(df[["title", "artist_name", "release", "year"]].head(10))
    print("\nMissing values:")
    print(df[["title", "artist_name", "release", "year"]].isna().sum())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} songs to {OUTPUT_FILE}")
