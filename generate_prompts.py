"""
8/26/26
AJ Boyd
"""

import pandas as pd
import json
from tqdm import tqdm
import numpy as np

# global constants
INPUT_FILE = "data/msd_split.csv"
OUTPUT_FILE = "data/song_prompts.csv"
TOP_N_TERMS = 10

# helper functions
def parse_json(s):
    try:
        return json.loads(s)
    except:
        return []

# get top N artist-associated terms based on weights
def get_top_terms(terms, weights, n=TOP_N_TERMS):
    terms = parse_json(terms)
    weights = parse_json(weights)

    if len(terms) != len(weights):
        return []

    pairs = sorted(
        zip(terms, weights),
        key=lambda x: x[1],
        reverse=True
    )

    return pairs[:n]

# build prompts based on artist terms and song metadata
def build_prompt(row):
    top_terms = get_top_terms(
        row["artist_terms"],
        row["artist_terms_weight"]
    )

    if top_terms:
        artist_terms = "\n".join(
            f"- {term} ({weight:.3f})"
            for term, weight in top_terms
        )
    else:
        artist_terms = "- None available"

    return f"""Generate 5–10 concise natural-language descriptors for this song.

The descriptors should characterize the song's likely mood, style, energy, atmosphere, and themes.

Artist-associated terms are contextual evidence and may not directly describe this specific song. Acoustic measurements should be treated as objective characteristics.

Song:
Title: {row["title"]}
Artist: {row["artist_name"]}

Artist-associated terms:
{artist_terms}

Acoustic characteristics:
Tempo: {row["tempo"]:.2f} BPM
Energy: {row["energy"]:.3f}
Loudness: {row["loudness"]:.2f} dB
Danceability: {row["danceability"]:.3f}
Duration: {row["duration"]:.2f} seconds
Key: {int(row["key"])}
Mode: {int(row["mode"])}
Time signature: {int(row["time_signature"])}

Return only one concise sentence (not a list) describing the song's overall musical character, incorporating its style, mood, energy, atmosphere, and other relevant characteristics inferred from the provided metadata and acoustic features. Don't include the song's name in the sentence.
For example: \"A fast, gritty garage-rock track with an energetic, rebellious character and a raw, driving atmosphere.\""""

# generate prompts for all songs in the dataset
if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)
    # df = df[df["split"] == "train"].copy()

    labels = np.load("data/cluster_labels.npy")

    if len(df) != len(labels):
        raise ValueError(
            f"Row mismatch: {len(df)} songs but {len(labels)} cluster labels."
        )

    df["cluster_label"] = labels
    prompts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating prompts"):
        prompts.append({
            "track_id": row.get("track_id", ""),
            "title": row["title"],
            "artist_name": row["artist_name"],
            "release": row["release"],
            "cluster_label": row["cluster_label"],
            "year": row["year"],
            "split": row["split"],
            "prompt": build_prompt(row)
        })

    prompts_df = pd.DataFrame(prompts)
    prompts_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(prompts_df)} prompts.")
    print(f"Saved to {OUTPUT_FILE}")