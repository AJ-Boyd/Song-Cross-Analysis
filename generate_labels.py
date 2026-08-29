"""
AJ Boyd
8/27/26

This script generates concise natural-language descriptors for songs in the Million Song Dataset using a language model. It reads song prompts from a CSV file, sends them to the model for processing, and saves the resulting descriptions along with metadata to a new CSV file. Additionally, it generates vector embeddings for the descriptions using a pre-trained sentence transformer model and saves them to disk.
(pre-req): run generate_prompts.py to create prompts for each song in the dataset, stored in data/song_prompts.csv
"""

import os
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

INPUT_FILE = "data/song_prompts.csv"
OUTPUT_FILE = "data/song_labels2.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

df = pd.read_csv(INPUT_FILE)
# df = df[df["split"] == "train"].copy()

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
Given the following information about a song, write exactly one sentence (not a list) that concisely describes the song's overall musical character, including its style, mood, energy, atmosphere, and other relevant characteristics.

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
            "cluster_label": row["cluster_label"],
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


LABEL_FILE = "data/song_labels.csv"
OUTPUT_EMBEDDINGS = "data/text_embeddings.npy"
OUTPUT_METADATA = "data/text_embedding_metadata.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

# load the dataset and filter out rows with missing track_id or semantic_description
df = pd.read_csv(LABEL_FILE)
df = df.dropna(subset=["track_id", "semantic_description"]).copy()

model = SentenceTransformer(MODEL_NAME)

embeddings = []

# encode each semantic description into a vector embedding
for description in tqdm(
    df["semantic_description"],
    total=len(df),
    desc="Generating text embeddings"
):
    embedding = model.encode(
        description,
        normalize_embeddings=True
    )
    embeddings.append(embedding)

embeddings = np.array(embeddings)

# save the embeddings and metadata to disk
np.save(OUTPUT_EMBEDDINGS, embeddings)

metadata = df[
    [
        "track_id",
        "title",
        "artist",
        "cluster_label",
        "split"
    ]
].copy()

metadata.to_csv(OUTPUT_METADATA, index=False)

print(f"Embedding shape: {embeddings.shape}")
print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")
print(f"Saved metadata to {OUTPUT_METADATA}")