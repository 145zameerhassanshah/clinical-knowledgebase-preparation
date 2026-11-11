
# embedding_evaluation/evaluate.py

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

from .dataset import PAIRS

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "evaluation_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "General": "sentence-transformers/all-MiniLM-L6-v2",
    "Biomedical": "NeuML/biomedbert-base-embeddings",
}

CATEGORIES = ["Synonyms", "Abbreviations", "Negation", "Numerical"]

records = [
    {
        "pair_id": pair_id,
        "category": category,
        "sentence_1": sentence_1,
        "sentence_2": sentence_2,
        "expected_equivalent": label,
    }
    for pair_id, category, sentence_1, sentence_2, label in PAIRS
]

results = pd.DataFrame(records)


def create_pca_plot(name, embeddings):
    """Project one model's 64 sentence vectors into 2D."""
    pca = PCA(n_components=2)
    points = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {
        "Synonyms": "green",
        "Abbreviations": "blue",
        "Negation": "red",
        "Numerical": "orange",
    }

    for i, pair in enumerate(PAIRS):
        pair_id, category, _, _, _ = pair
        p1 = points[2 * i]
        p2 = points[2 * i + 1]

        color = colors[category]

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            alpha=0.25,
            linewidth=1,
        )

        ax.scatter(
            p1[0], p1[1],
            color=color, marker="o", s=35
        )

        ax.scatter(
            p2[0], p2[1],
            color=color, marker="x", s=40
        )

    for category, color in colors.items():
        ax.scatter([], [], color=color, label=category)

    ax.set_title(f"{name} Model - PCA of Medical Sentence Pairs")
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"pca_{name.lower()}.png",
        dpi=180
    )
    plt.close(fig)


def evaluate_model(name, model_id):
    print(f"\nLoading {name}: {model_id}")

    model = SentenceTransformer(model_id, device="cpu")

    # Each pair contributes two sentences.
    sentences = []
    for pair in PAIRS:
        sentences.extend([pair[2], pair[3]])

    embeddings = model.encode(
        sentences,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    print("Embedding shape:", embeddings.shape)

    # Dot product equals cosine similarity for unit vectors.
    similarities = np.sum(
        embeddings[0::2] * embeddings[1::2],
        axis=1
    )

    results[f"{name.lower()}_cosine"] = similarities

    create_pca_plot(name, embeddings)

    # Save embeddings for reproducibility.
    np.save(
        OUTPUT / f"embeddings_{name.lower()}.npy",
        embeddings
    )

    del model, embeddings


def main():
    for name, model_id in MODELS.items():
        evaluate_model(name, model_id)

    results.to_csv(
        OUTPUT / "similarity_scores.csv",
        index=False
    )

    summary = results.groupby("category")[
        ["general_cosine", "biomedical_cosine"]
    ].agg(["mean", "std", "min", "max"])

    summary.to_csv(OUTPUT / "category_summary.csv")

    print("\n=== CATEGORY-WISE COMPARISON ===")
    print(summary.round(4).to_string())

    print("\n=== PAIR-WISE SCORES ===")
    print(
        results[
            ["pair_id", "category",
             "general_cosine", "biomedical_cosine"]
        ].round(4).to_string(index=False)
    )

    # Exploratory equivalent-vs-different discrimination.
    labels = results["expected_equivalent"]

    print("\n=== EXPLORATORY DISCRIMINATION ===")

    for name in ["general", "biomedical"]:
        scores = results[f"{name}_cosine"]

        positive_mean = scores[labels == 1].mean()
        negative_mean = scores[labels == 0].mean()
        separation = positive_mean - negative_mean
        auc = roc_auc_score(labels, scores)

        print(f"\n{name.upper()}")
        print(f"Equivalent mean: {positive_mean:.4f}")
        print(f"Different mean:  {negative_mean:.4f}")
        print(f"Mean separation: {separation:.4f}")
        print(f"Exploratory AUC: {auc:.4f}")

    print("\nResults saved in:", OUTPUT)


if __name__ == "__main__":
    main()
