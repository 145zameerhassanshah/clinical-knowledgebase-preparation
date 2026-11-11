
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "output" / "chunks.jsonl"
OUTPUT_FILE = ROOT / "output" / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk = json.loads(line)
                if chunk.get("text", "").strip():
                    chunks.append(chunk)

    return chunks


def generate_embeddings():
    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(OUTPUT_FILE, embeddings)

    print("Embedding generation complete!")
    print("Embedding shape:", embeddings.shape)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_embeddings()
