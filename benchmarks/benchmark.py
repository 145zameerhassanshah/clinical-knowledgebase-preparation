
"""
Day 4: Similarity Metrics & Vector Search Benchmark

Compare:
1. NumPy cosine similarity
2. NumPy dot product
3. NumPy Euclidean distance
4. FAISS exact search (IndexFlatIP)
5. FAISS approximate search (IndexIVFFlat)

Outputs:
- benchmark_results.csv
- accuracy_results.csv
- search_time_vs_dataset_size.png
- ivf_speed_recall_tradeoff.png
"""

from pathlib import Path
from time import perf_counter

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DIMENSIONS = 128
DATASET_SIZES = [10000, 25000, 50000]
NUM_QUERIES = 32
TOP_K = 10
REPEATS = 5
NPROBE_VALUES = [1, 4, 16, 32]

# Use one FAISS thread to reduce timing variability.
# NumPy may still use multithreaded BLAS.
faiss.omp_set_num_threads(1)

rng = np.random.default_rng(SEED)


# --------------------------------------------------
# DATA GENERATION
# --------------------------------------------------

def normalize(vectors):
    norms = np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    return vectors / np.maximum(norms, 1e-12)


def create_dataset(size):
    """
    Generate reproducible synthetic clustered vectors.

    This is NOT a real clinical dataset.
    """

    clusters = rng.normal(
        size=(100, DIMENSIONS)
    ).astype(np.float32)

    labels = rng.integers(
        0, 100, size=size
    )

    vectors = (
        clusters[labels]
        + rng.normal(
            scale=0.5,
            size=(size, DIMENSIONS)
        )
    ).astype(np.float32)

    query_labels = rng.integers(
        0, 100, size=NUM_QUERIES
    )

    queries = (
        clusters[query_labels]
        + rng.normal(
            scale=0.5,
            size=(NUM_QUERIES, DIMENSIONS)
        )
    ).astype(np.float32)

    return (
        np.ascontiguousarray(vectors),
        np.ascontiguousarray(queries)
    )


# --------------------------------------------------
# NUMPY SIMILARITY METHODS
# --------------------------------------------------

def top_k_from_scores(scores, k, largest=True):
    """
    Return sorted Top-K vector indices.

    Similarity: largest=True
    Distance:   largest=False
    """

    if largest:
        selected = np.argpartition(
            -scores, kth=k - 1, axis=1
        )[:, :k]

        selected_scores = np.take_along_axis(
            scores, selected, axis=1
        )

        order = np.argsort(
            -selected_scores, axis=1
        )

    else:
        selected = np.argpartition(
            scores, kth=k - 1, axis=1
        )[:, :k]

        selected_scores = np.take_along_axis(
            scores, selected, axis=1
        )

        order = np.argsort(
            selected_scores, axis=1
        )

    return np.take_along_axis(
        selected, order, axis=1
    )


def numpy_cosine_search(vectors, queries, k):
    """
    Cosine similarity with normalized vectors.
    Higher score = more similar.
    """

    scores = queries @ vectors.T

    return top_k_from_scores(
        scores, k, largest=True
    )


def numpy_dot_search(vectors, queries, k):
    """
    Raw dot product.
    Vector magnitude affects ranking.
    """

    scores = queries @ vectors.T

    return top_k_from_scores(
        scores, k, largest=True
    )


def numpy_euclidean_search(vectors, queries, k):
    """
    Squared Euclidean distance.

    Squared distance has the same ranking
    as Euclidean distance.
    """

    vector_norms = np.sum(
        vectors * vectors, axis=1
    )

    query_norms = np.sum(
        queries * queries, axis=1
    )

    distances = (
        query_norms[:, None]
        + vector_norms[None, :]
        - 2 * (queries @ vectors.T)
    )

    np.maximum(distances, 0, out=distances)

    return top_k_from_scores(
        distances, k, largest=False
    )


# --------------------------------------------------
# PERFORMANCE TIMING
# --------------------------------------------------

def measure_search(search_function):
    """
    Warm up, then time a batch of 32 queries.

    Returns median batch latency in milliseconds.
    """

    search_function()

    times = []

    for _ in range(REPEATS):
        start = perf_counter()

        search_function()

        elapsed = (
            perf_counter() - start
        ) * 1000

        times.append(elapsed)

    return float(np.median(times))


# --------------------------------------------------
# ACCURACY EVALUATION
# --------------------------------------------------

def recall_at_k(exact_ids, predicted_ids):
    """
    Average fraction of exact Top-K neighbors
    recovered by another search method.
    """

    recall_values = []

    for exact_row, predicted_row in zip(
        exact_ids, predicted_ids
    ):

        exact_set = set(
            int(value) for value in exact_row
        )

        predicted_set = set(
            int(value) for value in predicted_row
        )

        matches = len(
            exact_set.intersection(predicted_set)
        )

        recall_values.append(
            matches / len(exact_set)
        )

    return float(np.mean(recall_values))


# --------------------------------------------------
# MAIN BENCHMARK
# --------------------------------------------------

def run_benchmark():

    performance_records = []
    accuracy_records = []

    for size in DATASET_SIZES:

        print(f"\n{'=' * 55}")
        print(f"DATASET SIZE: {size:,}")
        print(f"{'=' * 55}")

        vectors, queries = create_dataset(size)

        normalized_vectors = np.ascontiguousarray(
            normalize(vectors).astype(np.float32)
        )

        normalized_queries = np.ascontiguousarray(
            normalize(queries).astype(np.float32)
        )

        # ------------------------------------------
        # NUMPY COSINE
        # ------------------------------------------

        cosine_fn = lambda: numpy_cosine_search(
            normalized_vectors,
            normalized_queries,
            TOP_K
        )

        cosine_time = measure_search(cosine_fn)
        cosine_ids = cosine_fn()

        # ------------------------------------------
        # NUMPY DOT PRODUCT
        # ------------------------------------------

        dot_fn = lambda: numpy_dot_search(
            vectors, queries, TOP_K
        )

        dot_time = measure_search(dot_fn)
        dot_ids = dot_fn()

        # ------------------------------------------
        # NUMPY EUCLIDEAN
        # ------------------------------------------

        euclidean_fn = lambda: numpy_euclidean_search(
            vectors, queries, TOP_K
        )

        euclidean_time = measure_search(
            euclidean_fn
        )

        euclidean_ids = euclidean_fn()

        methods = [
            ("NumPy Cosine", cosine_time),
            ("NumPy Dot", dot_time),
            ("NumPy Euclidean", euclidean_time),
        ]

        for method, elapsed in methods:
            performance_records.append({
                "dataset_size": size,
                "method": method,
                "search_time_ms": elapsed,
                "per_query_ms": elapsed / NUM_QUERIES,
                "build_time_ms": 0.0,
                "recall_at_10": np.nan,
            })

        print(f"NumPy Cosine:    {cosine_time:.3f} ms")
        print(f"NumPy Dot:       {dot_time:.3f} ms")
        print(f"NumPy Euclidean: {euclidean_time:.3f} ms")

        # ------------------------------------------
        # FAISS EXACT (INNER PRODUCT)
        # ------------------------------------------

        start = perf_counter()

        exact_index = faiss.IndexFlatIP(
            DIMENSIONS
        )

        exact_index.add(normalized_vectors)

        exact_build_time = (
            perf_counter() - start
        ) * 1000

        exact_fn = lambda: exact_index.search(
            normalized_queries, TOP_K
        )

        exact_time = measure_search(exact_fn)

        _, exact_ids = exact_fn()

        exact_agreement = recall_at_k(
            cosine_ids, exact_ids
        )

        performance_records.append({
            "dataset_size": size,
            "method": "FAISS Exact",
            "search_time_ms": exact_time,
            "per_query_ms": exact_time / NUM_QUERIES,
            "build_time_ms": exact_build_time,
            "recall_at_10": 1.0,
        })

        print(f"FAISS Exact:     {exact_time:.3f} ms")
        print(
            "NumPy cosine vs FAISS exact Recall@10:",
            f"{exact_agreement:.4f}"
        )

        accuracy_records.append({
            "dataset_size": size,
            "method": "FAISS Exact vs NumPy Cosine",
            "recall_at_10": exact_agreement,
        })

        # ------------------------------------------
        # FAISS APPROXIMATE IVF
        # ------------------------------------------

        nlist = min(
            256,
            max(32, int(np.sqrt(size)))
        )

        quantizer = faiss.IndexFlatIP(
            DIMENSIONS
        )

        ivf_index = faiss.IndexIVFFlat(
            quantizer,
            DIMENSIONS,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        train_count = min(size, 12000)

        train_ids = rng.choice(
            size,
            size=train_count,
            replace=False
        )

        training_vectors = np.ascontiguousarray(
            normalized_vectors[train_ids]
        )

        build_start = perf_counter()

        ivf_index.train(training_vectors)
        ivf_index.add(normalized_vectors)

        ivf_build_time = (
            perf_counter() - build_start
        ) * 1000

        print(
            f"IVF clusters: {nlist} | "
            f"Training vectors: {train_count}"
        )

        for nprobe in NPROBE_VALUES:

            ivf_index.nprobe = nprobe

            def ivf_fn():
                return ivf_index.search(
                    normalized_queries,
                    TOP_K
                )

            ivf_time = measure_search(ivf_fn)

            _, approximate_ids = ivf_fn()

            recall = recall_at_k(
                exact_ids, approximate_ids
            )

            name = f"FAISS IVF nprobe={nprobe}"

            performance_records.append({
                "dataset_size": size,
                "method": name,
                "search_time_ms": ivf_time,
                "per_query_ms": ivf_time / NUM_QUERIES,
                "build_time_ms": ivf_build_time,
                "recall_at_10": recall,
            })

            accuracy_records.append({
                "dataset_size": size,
                "method": name,
                "recall_at_10": recall,
            })

            print(
                f"{name}: "
                f"{ivf_time:.3f} ms | "
                f"Recall@10 = {recall:.4f}"
            )

        # Compare metric rankings on raw vectors.
        accuracy_records.extend([
            {
                "dataset_size": size,
                "method": "Raw Dot vs Normalized Cosine",
                "recall_at_10": recall_at_k(
                    cosine_ids, dot_ids
                ),
            },
            {
                "dataset_size": size,
                "method": "Raw L2 vs Normalized Cosine",
                "recall_at_10": recall_at_k(
                    cosine_ids, euclidean_ids
                ),
            },
        ])

    performance_df = pd.DataFrame(
        performance_records
    )

    accuracy_df = pd.DataFrame(
        accuracy_records
    )

    performance_df.to_csv(
        OUTPUT_DIR / "benchmark_results.csv",
        index=False
    )

    accuracy_df.to_csv(
        OUTPUT_DIR / "accuracy_results.csv",
        index=False
    )

    create_plots(performance_df)

    print("\nBENCHMARK COMPLETE")
    print("Results saved to:", OUTPUT_DIR)

    print("\n50,000 VECTOR RESULTS:")
    print(
        performance_df[
            performance_df["dataset_size"] == 50000
        ].to_string(index=False)
    )


# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

def create_plots(df):

    # Graph 1: Search time vs dataset size

    fig, ax = plt.subplots(figsize=(11, 7))

    for method, group in df.groupby("method"):

        group = group.sort_values(
            "dataset_size"
        )

        ax.plot(
            group["dataset_size"],
            group["search_time_ms"],
            marker="o",
            label=method
        )

    ax.set_title(
        "Vector Search Time vs Dataset Size"
    )

    ax.set_xlabel("Dataset Size (Vectors)")
    ax.set_ylabel(
        "Median Batch Search Time (ms, 32 queries)"
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "search_time_vs_dataset_size.png",
        dpi=180
    )

    plt.close(fig)

    # Graph 2: Speed vs Recall@10 at 50k

    subset = df[
        (df["dataset_size"] == 50000)
        & (df["method"].str.contains("FAISS IVF"))
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        subset["per_query_ms"],
        subset["recall_at_10"],
        marker="o"
    )

    for _, row in subset.iterrows():

        ax.annotate(
            row["method"].split("=")[-1],
            (
                row["per_query_ms"],
                row["recall_at_10"]
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel("Median Time Per Query (ms)")
    ax.set_ylabel("Recall@10")

    ax.set_title(
        "FAISS IVF Speed vs Recall (50,000 Vectors)"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "ivf_speed_recall_tradeoff.png",
        dpi=180
    )

    plt.close(fig)


if __name__ == "__main__":
    run_benchmark()
