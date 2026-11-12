
---

# DAY 4 — PERFORMANCE EVALUATION OF SIMILARITY METRICS AND VECTOR SEARCH ALGORITHMS

## 33. Task Overview

**Task Title:** Performance Evaluation of Similarity Metrics and Vector Search Algorithms

### Task Description

Implemented and compared cosine similarity, dot product, and Euclidean distance using NumPy. Developed a Top-K vector search system and evaluated its speed and retrieval accuracy against FAISS exact and approximate nearest-neighbor search algorithms.

The experiments were conducted using 10,000, 25,000, and 50,000 synthetic vectors. Search latency, index-building time, and Recall@10 were measured to analyze the trade-off between search speed and accuracy.

The task extends the existing Clinical Knowledgebase project by introducing performance benchmarking for large-scale vector retrieval.

### Objectives

- Understand vector similarity metrics.
- Implement cosine similarity using NumPy.
- Implement dot product using NumPy.
- Implement Euclidean distance using NumPy.
- Develop a Top-K similarity search algorithm.
- Implement FAISS exact nearest-neighbor search.
- Implement FAISS approximate nearest-neighbor search.
- Evaluate at least 50,000 vectors.
- Compare search latency and index-building time.
- Evaluate approximate search using Recall@K.
- Visualize search performance and accuracy trade-offs.

---

## 34. Real-World Problem

Clinical AI and Retrieval-Augmented Generation (RAG) systems may need to search thousands or millions of document embeddings.

Exact search compares a query against every stored vector to identify the nearest neighbors. While exact search provides a reference result, exhaustive comparison can become computationally expensive as datasets grow.

Approximate nearest-neighbor algorithms aim to reduce search time by examining selected candidate vectors instead of the entire dataset.

This experiment investigates:

**How can a vector search system retrieve relevant information quickly while maintaining acceptable nearest-neighbor recall?**

The experiment uses synthetic vectors to evaluate search algorithms independently of embedding-model quality.

---

## 35. Similarity Metrics

### 35.1 Cosine Similarity

Cosine similarity measures the directional similarity between two vectors.

Formula:

cosine(A, B) = (A · B) / (||A|| × ||B||)

Higher cosine similarity indicates greater directional alignment.

Cosine similarity is commonly used for semantic text retrieval.

### 35.2 Dot Product

Dot product calculates the sum of element-wise products between two vectors.

Formula:

A · B = Σ(Ai × Bi)

Unlike cosine similarity, raw dot product is affected by vector magnitude.

For L2-normalized vectors, dot product equals cosine similarity.

### 35.3 Euclidean Distance

Euclidean distance measures the straight-line distance between vectors.

Formula:

d(A, B) = sqrt(Σ(Ai - Bi)²)

Lower distance indicates closer vectors.

The implementation uses squared Euclidean distance for ranking because it produces the same ranking as ordinary Euclidean distance without computing square roots.

### 35.4 Comparison

| Metric | Ranking Direction | Main Property |
|---|---|---|
| Cosine similarity | Higher is better | Directional similarity |
| Dot product | Higher is better | Magnitude-sensitive similarity |
| Euclidean distance | Lower is better | Geometric distance |

---

## 36. Search Algorithms

### 36.1 NumPy Exact Search

Implemented vectorized Top-K retrieval using NumPy.

The implementation calculates similarity or distance scores between query vectors and database vectors, then selects the Top-K results.

Implemented methods:

- NumPy cosine similarity
- NumPy dot product
- NumPy Euclidean distance

### 36.2 FAISS Exact Search

Used FAISS `IndexFlatIP` with normalized vectors.

This performs exhaustive inner-product search and provides exact nearest neighbors for the chosen metric.

FAISS exact results were also compared against NumPy normalized cosine search.

### 36.3 FAISS Approximate Search

Used FAISS `IndexIVFFlat`.

This algorithm divides vectors into clusters and searches selected inverted lists.

Important parameters:

- **nlist:** Number of inverted lists or clusters.
- **nprobe:** Number of inverted lists searched for a query.

Increasing nprobe generally improves the chance of recovering exact nearest neighbors but also increases search computation.

Tested nprobe values:

- 1
- 4
- 16
- 32

---

## 37. Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset type | Synthetic clustered vectors |
| Dataset sizes | 10,000 / 25,000 / 50,000 |
| Vector dimensions | 128 |
| Queries per dataset | 32 |
| Top-K | 10 |
| Repeated timing runs | 5 |
| Timing statistic | Median after warm-up |
| FAISS exact index | IndexFlatIP |
| FAISS approximate index | IndexIVFFlat |
| IVF nprobe | 1, 4, 16, 32 |
| Random seed | 42 |

Synthetic data was used to create a reproducible benchmarking workload.

This dataset is separate from the clinical text embeddings generated during Day 1 and Day 2.

---

## 38. Actual Benchmark Results

All results below were recorded by executing the benchmark locally using Python.

Search time represents the median processing time for a batch of 32 queries.

### 38.1 Search Time Comparison

| Method | 10,000 Vectors | 25,000 Vectors | 50,000 Vectors |
|---|---:|---:|---:|
| NumPy Cosine | 12.939 ms | 31.798 ms | 68.471 ms |
| NumPy Dot | 11.812 ms | 30.846 ms | 65.222 ms |
| NumPy Euclidean | 21.711 ms | 55.210 ms | 124.808 ms |
| FAISS Exact | 119.073 ms | 186.722 ms | 376.940 ms |
| FAISS IVF nprobe=1 | 1.187 ms | 2.232 ms | 3.877 ms |
| FAISS IVF nprobe=4 | 4.214 ms | 6.065 ms | 9.214 ms |
| FAISS IVF nprobe=16 | 13.391 ms | 21.252 ms | 30.054 ms |
| FAISS IVF nprobe=32 | 26.527 ms | 41.268 ms | 58.213 ms |

### 38.2 Recall@10 Comparison

Recall@10 measures the fraction of exact Top-10 neighbors recovered by approximate search.

| IVF Configuration | 10,000 | 25,000 | 50,000 |
|---|---:|---:|---:|
| nprobe=1 | 94.06% | 86.25% | 73.44% |
| nprobe=4 | 100% | 100% | 100% |
| nprobe=16 | 100% | 100% | 100% |
| nprobe=32 | 100% | 100% | 100% |

### 38.3 Exact Search Validation

FAISS exact search matched NumPy normalized cosine Top-K results.

Observed Recall@10 agreement:

**1.0000 (100%)**

This validates agreement between the two exact-search implementations on the tested datasets.

---

## 39. Key Findings

### Finding 1 — Fastest Search

At 50,000 vectors, FAISS IVF with nprobe=1 achieved the lowest measured latency:

**3.877 ms**

However, Recall@10 was only:

**73.44%**

This shows that faster approximate search can miss exact nearest neighbors.

### Finding 2 — Best Observed Speed–Recall Trade-Off

At 50,000 vectors, FAISS IVF with nprobe=4 achieved:

- Search time: **9.214 ms**
- Recall@10: **100%**

It achieved full recall on the tested synthetic workload while being substantially faster than the measured exhaustive-search methods.

This result is dataset-specific and does not guarantee full recall on other data.

### Finding 3 — Effect of nprobe

Increasing nprobe improved approximate search recall in the experiments.

However, higher nprobe values also increased query latency.

This demonstrates the trade-off between approximate-search speed and nearest-neighbor accuracy.

### Finding 4 — NumPy vs FAISS Exact

At 50,000 vectors:

- NumPy cosine: 68.471 ms
- FAISS exact: 376.940 ms

FAISS exact was slower in this specific experiment.

This does not establish that FAISS is generally slower than NumPy.

The benchmark used a one-thread FAISS configuration without explicitly matching NumPy BLAS threading. Hardware, threading, batch size, and implementation details influence performance.

---

## 40. Generated Performance Visualizations

The benchmark produces two primary graphs.

### 40.1 Search Time vs Dataset Size

File:

`benchmark_results/search_time_vs_dataset_size.png`

This graph shows how query latency changes as the vector database increases from 10,000 to 50,000 vectors.

### 40.2 Speed–Recall Trade-Off

File:

`benchmark_results/ivf_speed_recall_tradeoff.png`

This graph compares FAISS IVF query latency and Recall@10 at 50,000 vectors for different nprobe values.

### Result Files

```text
benchmark_results/
├── benchmark_results.csv
├── accuracy_results.csv
├── search_time_vs_dataset_size.png
└── ivf_speed_recall_tradeoff.png
```

---

## 41. Day 4 Implementation Files

```text
benchmarks/
└── benchmark.py

pages/
└── 4_Vector_Search_Performance.py

benchmark_results/
├── benchmark_results.csv
├── accuracy_results.csv
├── search_time_vs_dataset_size.png
└── ivf_speed_recall_tradeoff.png
```

`benchmarks/benchmark.py` implements the computational benchmark.

The Streamlit page provides an interface for exploring benchmark results and visualizations when installed.

---

## 42. Installation and Execution

Activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install required dependencies:

```powershell
python -m pip install numpy pandas matplotlib faiss-cpu streamlit
```

Execute the Day 4 benchmark:

```powershell
python benchmarks/benchmark.py
```

The script generates the benchmark CSV files and performance graphs.

### Run Streamlit

Configure Hugging Face cache if using the clinical embedding functionality:

```powershell
$env:HF_HOME = "D:\huggingface_cache"
$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"
$env:HF_XET_CACHE = "D:\huggingface_cache\xet"
```

Launch the application:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

http://127.0.0.1:8501

Select the Vector Search Performance page from the sidebar if installed.

---

## 43. Testing and Verification

### CLI Testing

The following have been successfully executed:

- NumPy cosine similarity search
- NumPy dot-product search
- NumPy Euclidean-distance search
- FAISS exact search
- FAISS IVF approximate search
- Benchmarks at 10,000 vectors
- Benchmarks at 25,000 vectors
- Benchmarks at 50,000 vectors
- Recall@10 calculations
- Performance CSV generation
- Performance graph generation

### Browser Testing

The Streamlit server was started locally.

However, Windows asyncio connection-reset messages (`WinError 10054`) were observed.

Full browser functionality and guided end-to-end workflow testing remain pending.

The successful CLI benchmark is independent of this browser issue.

---

## 44. Project Limitations

- Synthetic vectors are not real clinical document embeddings.
- Benchmark results depend on dataset distribution.
- Search performance depends on hardware and threading.
- FAISS and NumPy threading were not fully matched.
- Recall@10 measures agreement with exact neighbors, not clinical relevance.
- Index training and building costs must be considered separately from query latency.
- Further evaluation is needed on realistic embedding datasets.

---

## 45. Future Improvements

- Fair benchmarking with controlled CPU thread counts.
- Additional random seeds and repeated trials.
- Larger datasets: 100,000 and 1,000,000 vectors.
- Evaluation using actual clinical text embeddings.
- HNSW and other approximate indexing methods.
- Automated search algorithm recommendations.
- Configurable benchmark options through Streamlit.
- Retrieval quality evaluation using labeled clinical queries.

---

## 46. Day 4 Key Deliverables

1. NumPy similarity metrics implementation.
2. Top-K exact vector search.
3. FAISS IndexFlatIP implementation.
4. FAISS IndexIVFFlat implementation.
5. 50,000-vector performance benchmark.
6. Search-time comparison across dataset sizes.
7. Recall@10 evaluation.
8. Performance visualizations.
9. Benchmark result CSV files.
10. Integration with the existing Clinical Knowledgebase project.

---

## 47. My Contribution and AI Assistance

### My Contribution

I completed the Performance Evaluation of Similarity Metrics and Vector Search Algorithms task using Python and VS Code.

I executed NumPy and FAISS benchmarks using 10,000, 25,000, and 50,000 vectors.

I evaluated search latency, compared exact and approximate retrieval, reviewed Recall@10 results, and generated performance graphs.

I also worked on integrating the evaluation into my existing Clinical Knowledgebase project and documenting the experimental findings.

### AI Assistance

I used ChatGPT to understand similarity metrics, vector search algorithms, FAISS indexing, approximate nearest-neighbor search, and Recall@K.

ChatGPT assisted with Python code development, debugging, experimental design, Streamlit integration, and project documentation.

The benchmark code was executed locally, and the reported results were reviewed by me.

### Tools Used

- Python
- VS Code
- NumPy
- FAISS
- Pandas
- Matplotlib
- Streamlit
- Git / GitHub
- ChatGPT

### AI Usage Summary

- Human work and decisions: **60%**
- AI assistance: **40%**

The percentages are self-assessed estimates.

---

## 48. Final Conclusion

The Day 4 experiment demonstrated the trade-off between exact and approximate vector search.

Exact search provides a reliable nearest-neighbor reference, while approximate search can improve retrieval speed by searching fewer candidates.

At 50,000 vectors, FAISS IVF with nprobe=4 achieved 100% Recall@10 with 9.214 ms median batch search time on the tested synthetic dataset.

The results show why similarity metrics, indexing algorithms, search latency, and retrieval recall should all be evaluated when designing scalable AI retrieval systems.

Together, Day 1 through Day 4 provide an educational foundation for clinical document processing, embedding evaluation, semantic retrieval, and vector search performance optimization.

**Disclaimer:** This project is intended for research and educational purposes and is not a validated clinical decision-support system.
