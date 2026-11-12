"""Day 4 Streamlit page: performance results, diagnostics and vector playground.
Place at <repository>/pages/4_Vector_Search_Performance.py.
Launch existing root app: python -m streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vector Search Lab", page_icon="⚡", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
SCRIPT = ROOT / "benchmarks" / "benchmark.py"

st.title("⚡ Vector Search Performance Lab")
st.caption("Day 4 · NumPy vs FAISS · Exact and approximate retrieval")
st.info("Benchmark uses synthetic vectors. Search accuracy here means agreement with exact nearest neighbors, **not** clinical relevance or medical safety.")


def read_results():
    """Support both benchmark CSV layouts used in the project."""
    paths = [RESULTS / "benchmark_results.csv", ROOT / "benchmarks" / "results" / "benchmark_results.csv"]
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            st.error(f"Could not read {path}: {exc}")
            return None, path
        if "search_time_ms" not in df.columns and "ms_per_query" in df.columns:
            # Older benchmark records per-query rather than batch timing.
            df["per_query_ms"] = df["ms_per_query"]
        if "per_query_ms" not in df.columns and "search_time_ms" in df.columns:
            # Only safe for the known 32-query benchmark configuration.
            df["per_query_ms"] = df["search_time_ms"] / 32
        if "recall_at_10" not in df.columns and "recall_at_k" in df.columns:
            df["recall_at_10"] = df["recall_at_k"]
        if "build_time_ms" not in df.columns and "build_ms" in df.columns:
            df["build_time_ms"] = df["build_ms"]
        required = {"dataset_size", "method", "per_query_ms"}
        if missing := required - set(df.columns):
            st.error(f"CSV is missing required fields: {sorted(missing)}")
            return None, path
        return df, path
    return None, None


def diagnostics():
    checks = {
        "Benchmark script": SCRIPT.exists(),
        "Results CSV": any(p.exists() for p in (RESULTS / "benchmark_results.csv", ROOT / "benchmarks" / "results" / "benchmark_results.csv")),
        "NumPy": True,
        "Streamlit": True,
    }
    try:
        import faiss  # noqa: F401
        checks["FAISS"] = True
    except ImportError:
        checks["FAISS"] = False
    return checks


tab_results, tab_play, tab_run, tab_help = st.tabs([
    "📊 Results & charts", "🔎 Live vector search", "▶️ Run benchmark", "🛠️ Setup & diagnostics"
])

with tab_results:
    df, data_path = read_results()
    if df is None:
        st.warning("No benchmark results found. Run the existing Day 4 benchmark first or use the Run benchmark tab.")
    else:
        st.success(f"Loaded: {data_path.relative_to(ROOT)}")
        sizes = sorted(df["dataset_size"].dropna().astype(int).unique().tolist())
        selected_size = st.selectbox("Dataset size for detailed comparison", sizes, index=len(sizes)-1)
        selected = df.loc[df["dataset_size"] == selected_size].copy()
        selected["per_query_ms"] = pd.to_numeric(selected["per_query_ms"], errors="coerce")
        if "recall_at_10" in selected:
            selected["recall_at_10"] = pd.to_numeric(selected["recall_at_10"], errors="coerce")

        c1, c2, c3 = st.columns(3)
        c1.metric("Vectors", f"{selected_size:,}")
        c2.metric("Search methods", len(selected))
        fastest = selected.sort_values("per_query_ms").iloc[0]
        c3.metric("Lowest measured latency", f"{fastest['per_query_ms']:.3f} ms/query", fastest["method"], delta_color="off")

        st.subheader("Search speed versus dataset size")
        chart_df = df.pivot_table(index="dataset_size", columns="method", values="per_query_ms", aggfunc="mean")
        st.line_chart(chart_df, x_label="Number of vectors", y_label="Measured milliseconds / query")
        st.caption("These are recorded measurements on the original machine. Results can vary with CPU, threading, query batches and dataset distribution.")

        st.subheader(f"Method comparison — {selected_size:,} vectors")
        st.bar_chart(selected.set_index("method")["per_query_ms"], y_label="Milliseconds / query")
        if "recall_at_10" in selected.columns:
            ivf = selected[selected["method"].str.contains("IVF", case=False, na=False)].copy()
            if not ivf.empty:
                st.subheader("Approximate search: latency and Recall@10")
                st.dataframe(ivf[["method", "per_query_ms", "recall_at_10"]].sort_values("per_query_ms"), hide_index=True, use_container_width=True)
                plot_df = ivf[["per_query_ms", "recall_at_10", "method"]].dropna()
                st.scatter_chart(plot_df, x="per_query_ms", y="recall_at_10", color="method", x_label="ms/query", y_label="Recall@10")
                perfect = ivf[ivf["recall_at_10"] >= 0.999999]
                if not perfect.empty:
                    best = perfect.sort_values("per_query_ms").iloc[0]
                    st.info(f"Fastest IVF configuration with measured Recall@10 = 1.0 at this size: **{best['method']}**. This is dataset-specific, not a guarantee.")
        st.subheader("Full results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download benchmark CSV", df.to_csv(index=False).encode("utf-8"), file_name="vector_benchmark_results.csv", mime="text/csv")
        plots = [
            (RESULTS / "search_time_vs_dataset_size.png", "Search time vs dataset size"),
            (RESULTS / "ivf_speed_recall_tradeoff.png", "IVF speed–recall trade-off"),
            (ROOT / "benchmarks" / "results" / "search_time_vs_dataset_size.png", "Search time vs dataset size (alternate output)"),
            (ROOT / "benchmarks" / "results" / "ivf_speed_recall_tradeoff.png", "IVF speed–recall trade-off (alternate output)"),
        ]
        existing = [(p, t) for p, t in plots if p.exists()]
        if existing:
            st.subheader("Saved benchmark diagrams")
            for path, label in existing:
                st.markdown(f"**{label}**")
                st.image(str(path), use_container_width=True)


@st.cache_resource(show_spinner="Generating demonstration vectors and building FAISS indexes...")
def playground_indices(size: int, dimensions: int):
    import faiss
    rng = np.random.default_rng(2026)
    # Synthetic vectors are normalized for cosine/IP equivalence.
    centers = rng.normal(size=(64, dimensions)).astype("float32")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.integers(0, 64, size=size)
    vectors = centers[labels] + 0.3 * rng.normal(size=(size, dimensions)).astype("float32")
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    exact = faiss.IndexFlatIP(dimensions)
    exact.add(vectors)
    nlist = min(128, max(16, int(np.sqrt(size))))
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimensions), dimensions, nlist, faiss.METRIC_INNER_PRODUCT)
    training = vectors[rng.choice(size, size=min(size, 10000), replace=False)]
    ivf.train(np.ascontiguousarray(training))
    ivf.add(vectors)
    return vectors, exact, ivf, nlist


with tab_play:
    st.subheader("Try a real vector query")
    st.write("Generate a synthetic vector database and see how exact and approximate search rank the **same query**.")
    c1, c2, c3 = st.columns(3)
    size = c1.selectbox("Number of vectors", [1000, 5000, 10000], index=1)
    k = c2.slider("Top-K", 1, 20, 10)
    nprobe = c3.slider("IVF nprobe", 1, 32, 4)
    query_row = st.number_input("Choose a vector ID as the reference query", 0, size-1, 42, step=1)
    if st.button("Search vectors", type="primary"):
        try:
            import faiss
            vectors, flat, ivf, nlist = playground_indices(size, 128)
            query = vectors[int(query_row):int(query_row)+1].copy()
            # Note: exact query is one of the stored vectors, so top-1 should match itself.
            start = perf_counter()
            numpy_scores = query @ vectors.T
            numpy_ids = np.argsort(-numpy_scores[0])[:k]
            numpy_ms = (perf_counter()-start)*1000
            start = perf_counter()
            exact_scores, exact_ids = flat.search(query, k)
            exact_ms = (perf_counter()-start)*1000
            ivf.nprobe = nprobe
            start = perf_counter()
            ivf_scores, ivf_ids = ivf.search(query, k)
            ivf_ms = (perf_counter()-start)*1000
            matched = len(set(map(int, exact_ids[0])) & set(map(int, ivf_ids[0]))) / k
            st.success(f"Search finished · IVF clusters: {nlist} · IVF Recall@{k}: {matched:.1%}")
            st.caption("Single-query timings below are illustrative and unstable. Use the batch benchmark for reliable performance comparisons.")
            st.dataframe(pd.DataFrame({"method": ["NumPy Cosine", "FAISS Exact", f"FAISS IVF (nprobe={nprobe})"], "single_query_ms": [numpy_ms, exact_ms, ivf_ms]}), hide_index=True, use_container_width=True)
            a, b = st.columns(2)
            with a:
                st.markdown("**FAISS exact Top-K**")
                st.dataframe(pd.DataFrame({"vector_id": exact_ids[0], "cosine_score": exact_scores[0]}), hide_index=True)
            with b:
                st.markdown("**FAISS IVF Top-K**")
                st.dataframe(pd.DataFrame({"vector_id": ivf_ids[0], "cosine_score": ivf_scores[0]}), hide_index=True)
            if set(map(int, numpy_ids)) != set(map(int, exact_ids[0])):
                st.warning("NumPy/FAISS exact IDs differ. Check float32 precision and equal-score ties.")
            st.caption("The selected query is one of the indexed synthetic vectors, so ID-matching is easier than for an independent unseen query. This playground is for inspection, not rigorous benchmark accuracy.")
        except ImportError:
            st.error("FAISS is missing. Install it with: python -m pip install faiss-cpu")
        except Exception as exc:
            st.error(f"Search failed: {exc}")

with tab_run:
    st.subheader("Run the complete benchmark")
    st.write("Executes your existing `benchmarks/benchmark.py` and refreshes the CSV and plots. It tests 10k, 25k and 50k vectors.")
    if not SCRIPT.exists():
        st.error("Missing benchmarks/benchmark.py. Copy your existing Day 4 benchmark file into that path.")
    else:
        st.warning("This is a CPU-intensive local experiment. Run it only when you want to regenerate results. The application may be unresponsive until the command finishes.")
        if st.button("Run 50,000-vector benchmark", type="primary"):
            with st.spinner("Executing benchmark in the current Python environment..."):
                try:
                    # Run a local, fixed script (not user-supplied shell commands).
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT)],
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=900,
                        check=False,
                        env=os.environ.copy(),
                    )
                    output = (result.stdout or "") + "\n" + (result.stderr or "")
                    st.code(output[-25000:], language="text")
                    if result.returncode == 0:
                        st.success("Benchmark command completed. Open Results & charts to view the new files.")
                    else:
                        st.error(f"Benchmark failed with exit code {result.returncode}. Inspect the log above.")
                except subprocess.TimeoutExpired:
                    st.error("Benchmark exceeded the 15-minute limit. Try running it directly in a terminal to diagnose resource constraints.")
                except Exception as exc:
                    st.error(f"Could not launch benchmark: {exc}")

with tab_help:
    st.subheader("Environment and project diagnostics")
    st.code(f"Project: {ROOT}\nPython: {sys.executable}\nBenchmark: {SCRIPT}", language="text")
    for item, is_ok in diagnostics().items():
        st.write(("✅" if is_ok else "❌") + " " + item)
    st.markdown("**If a package is missing**, activate the project's environment and install:")
    st.code("python -m pip install streamlit numpy pandas matplotlib faiss-cpu", language="powershell")
    st.markdown("**If results are missing**, run:")
    st.code("python benchmarks/benchmark.py", language="powershell")
    st.markdown("**Run the existing multipage app:**")
    st.code("python -m streamlit run app.py", language="powershell")
    st.info("This page does not modify your clinical documents, existing indexes, Day 3 datasets or Git branches. It only reads benchmark results and optionally runs the fixed local benchmark script.")
