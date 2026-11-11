
from sentence_transformers import SentenceTransformer
import numpy as np

MODELS = {
    "General": "sentence-transformers/all-MiniLM-L6-v2",
    "Biomedical": "NeuML/biomedbert-base-embeddings",
}

SENTENCES = [
    "The patient has hypertension.",
    "The patient has high blood pressure.",
    "The patient does not have hypertension."
]


def compare_models():
    for name, model_id in MODELS.items():

        print(f"\nLoading {name} model...")
        print(f"Model ID: {model_id}")

        model = SentenceTransformer(model_id, device="cpu")

        embeddings = model.encode(
            SENTENCES,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        print(f"\nModel: {name}")
        print("Embedding shape:", embeddings.shape)

        synonym_similarity = np.dot(
            embeddings[0],
            embeddings[1]
        )

        negation_similarity = np.dot(
            embeddings[0],
            embeddings[2]
        )

        print(
            "Synonym similarity:",
            round(float(synonym_similarity), 4)
        )

        print(
            "Negation similarity:",
            round(float(negation_similarity), 4)
        )


if __name__ == "__main__":
    compare_models()
