import os
import pandas as pd
import requests
from tqdm import tqdm

INPUT_FILE = "data/song_prompts.csv"
OUTPUT_FILE = "data/song_labels.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

df = pd.read_csv(INPUT_FILE)
df = df[df["split"] == "train"].copy()

if os.path.exists(OUTPUT_FILE):
    results = pd.read_csv(OUTPUT_FILE)
    completed = set(results["track_id"].astype(str))
else:
    results = pd.DataFrame(columns=[
        "track_id",
        "title",
        "artist",
        "cluster_label",
        "split",
        "semantic_description"
    ])
    completed = set()

remaining = df[~df["track_id"].astype(str).isin(completed)]

print(f"Training songs: {len(df)}")
print(f"Already labeled: {len(completed)}")
print(f"Remaining: {len(remaining)}")

for _, row in tqdm(
    remaining.iterrows(),
    total=len(remaining),
    desc="Generating labels"
):
    prompt = f"""
Given the following information about a song, write exactly one sentence
describing its overall musical character.

Use only characteristics reasonably supported by the provided information.
Do not mention numerical values, the analysis process, or the information
provided to you. Do not invent specific instruments or lyrical themes.

Return only the sentence.

{row["prompt"]}
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        description = response.json()["response"].strip()

        result = {
            "track_id": row["track_id"],
            "title": row["title"],
            "artist": row["artist_name"],
            "split": row["split"],
            "semantic_description": description
        }

        results = pd.concat(
            [results, pd.DataFrame([result])],
            ignore_index=True
        )

        results.to_csv(OUTPUT_FILE, index=False)

    except Exception as e:
        print(f"\nError processing {row['track_id']}: {e}")

print(f"\nSaved labels to {OUTPUT_FILE}")