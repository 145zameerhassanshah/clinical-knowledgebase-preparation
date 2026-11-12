# Clinical AI Research Workbench

**Clinical Knowledgebase Preparation, Semantic Search, Biomedical Embedding Evaluation & Vector Search Performance Benchmarking**

A Python and Streamlit-based AI research project for processing clinical documents, preparing retrieval-ready knowledgebases, implementing vector-based semantic search, evaluating biomedical embedding models, and benchmarking exact and approximate nearest-neighbor search algorithms.

The project combines four practical AI/ML engineering tasks into a single repository and provides a foundation for future Retrieval-Augmented Generation (RAG) research.

**Repository:** https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation

---

## 1. Project Overview

Clinical guidelines and healthcare documents contain complex terminology, abbreviations, numerical measurements, treatment instructions, and important contextual differences.

Building an effective clinical information retrieval system requires reliable document processing, vector embeddings, semantic search, and evaluation of retrieval performance.

This project was developed through four connected learning tasks.

| Stage | Task | Main Objective |
|---|---|---|
| Day 1 | Clinical Knowledgebase Preparation | Convert clinical documents into structured, metadata-rich chunks |
| Day 2 | Vector Databases & Semantic Search | Generate embeddings and search clinical documents using ChromaDB and FAISS |
| Day 3 | Embedding Model Evaluation | Compare general-purpose and biomedical embeddings using medical sentence pairs |
| Day 4 | Similarity Metrics & Vector Search Performance | Compare NumPy and FAISS exact/approximate search on up to 50,000 vectors |

### Main Technologies

- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Sentence Transformers
- Hugging Face
- ChromaDB
- FAISS
- Matplotlib
- PyMuPDF
- python-docx
- BeautifulSoup
- Tesseract OCR
- Git and GitHub

---

## 2. Real-World Problem

Healthcare organizations manage large collections of clinical practice guidelines, research papers, hospital protocols, treatment recommendations, and standard operating procedures.

Traditional keyword search may struggle when different terms describe the same medical concept.

**Example:**

Document text: "The patient presents with dyspnea."

Search query: "The patient has shortness of breath."

Both statements describe closely related clinical meaning despite different wording.

However, medical information retrieval must also distinguish clinically important differences.

**Negation example:**

- "The patient has pneumonia."
- "The patient does not have pneumonia."

**Numerical difference example:**

- "The patient's potassium level is 4.0 mmol/L."
- "The patient's potassium level is 6.5 mmol/L."

These statements share many words but have different clinical meanings.

A reliable retrieval system should recognize synonyms and abbreviations while avoiding incorrect assumptions about negation, numerical values, and other critical information.

The project also addresses the computational challenge of finding relevant information efficiently when a vector database contains tens of thousands of embeddings.

### Project Objectives

- Process clinical documents from multiple file formats.
- Extract and organize medical text.
- Generate logical sections and retrieval-ready chunks.
- Preserve source metadata.
- Convert document chunks into numerical embeddings.
- Implement semantic search using ChromaDB and FAISS.
- Compare general-purpose and biomedical embedding models.
- Investigate synonyms, abbreviations, negation, and numerical differences.
- Implement cosine similarity, dot product, and Euclidean distance.
- Compare exact and approximate vector search.
- Evaluate latency, Recall@K, and indexing performance.
- Visualize results through charts and browser dashboards.

**Disclaimer:** This is an educational research prototype, not a validated medical decision-support system.

---

## 3. System Architecture

### 3.1 Clinical Document Processing and Retrieval

**Clinical Documents → Upload → File Detection → Text Extraction/OCR → Section Detection → Chunking → Metadata → Validation → Embeddings → ChromaDB/FAISS → Semantic Search → Top-K Results**

The processing workflow creates a retrieval-ready dataset containing clinical passages and their original source information.

The retrieval system converts queries into embeddings and finds related stored document chunks.

### 3.2 Day 3 — Embedding Model Evaluation

**32 Medical Sentence Pairs → General MiniLM + Biomedical BiomedBERT → Embeddings → Cosine Similarity → Category Analysis → PCA → Research Findings**

This is an independent benchmark using a predefined medical sentence-pair dataset.

### 3.3 Day 4 — Vector Search Performance

**Synthetic Dataset → NumPy Similarity Metrics → FAISS Exact Search → FAISS IVF Approximate Search → Latency + Recall@10 → CSV Results + Performance Charts**

The Day 4 benchmark uses synthetic numerical vectors rather than real clinical documents.

### 3.4 Unified Browser Application

The Streamlit application supports a modular workflow:

- Clinical Knowledgebase Builder
- Vector Database and Semantic Search
- Biomedical Embedding Evaluation
- Vector Search Performance Dashboard
- Optional Guided Clinical Workflow

The modules share a repository but do not all automatically evaluate the same uploaded dataset.

---

# DAY 1 — CLINICAL KNOWLEDGEBASE PREPARATION

## 4. Task Description

**Task Title:** Clinical Knowledgebase Preparation

Develop a Python-based preprocessing pipeline to transform clinical documents into structured, retrieval-ready datasets while preserving logical sections and source metadata.

### Key Deliverables

- Multi-format clinical document ingestion.
- Text extraction and OCR support.
- Logical section detection.
- Structure-aware text chunking.
- Metadata preservation.
- Knowledgebase validation.
- JSON and CSV output generation.
- Streamlit document-upload interface.

## 5. Supported Document Formats

The application supports:

- PDF
- DOCX
- TXT
- HTML / HTM
- PNG
- JPG / JPEG
- TIFF
- BMP
- WEBP

### Text Extraction Technologies

| Format | Technology |
|---|---|
| Digital PDFs | PyMuPDF |
| Word documents | python-docx |
| HTML documents | BeautifulSoup |
| Scanned documents/images | Tesseract OCR |
| Tabular inspection | Pandas |

Tesseract OCR requires a separate system installation.

## 6. Knowledgebase Processing Pipeline

### 6.1 Document Ingestion

The application accepts uploaded clinical files and prepares them for text extraction.

### 6.2 Text Extraction

Extracts readable text from the supported document formats.

### 6.3 Logical Section Detection

Identifies meaningful sections where possible, including:

- Assessment
- Diagnosis
- Treatment
- Monitoring
- Follow-up
- Contraindications
- Emergency Management

Section detection uses structural heuristics.

### 6.4 Structure-Aware Chunking

Extracted text is divided into retrieval-sized chunks while attempting to preserve section and sentence boundaries.

### 6.5 Metadata Preservation

Each chunk retains information such as:

- Chunk ID
- Document ID
- Section ID
- Section title
- Chunk index
- Source filename
- Source type
- Extraction method
- Original text

Metadata supports traceability between retrieved passages and their original documents.

## 7. Day 1 Validation Results

The processing pipeline was tested through the command-line interface with two clinical guideline documents.

| Metric | Recorded Result |
|---|---:|
| Documents discovered | 2 |
| Documents processed | 2 |
| Logical sections | 168 |
| Chunks generated | 631 |
| Failed documents | 0 |
| Average chunk size | 720.64 characters |
| Retrieval preparation status | READY |

### Generated Files

- `output/documents.json`
- `output/sections.json`
- `output/chunks.jsonl`
- `output/chunks.csv`
- `output/build_report.json`

### Day 1 Outcome

Successfully processed two clinical guidelines and generated 631 retrieval-ready text chunks.

These results establish that the preprocessing pipeline executed successfully. They do not independently establish clinical retrieval accuracy.

---

# DAY 2 — VECTOR DATABASES & SEMANTIC SEARCH

## 8. Task Description

**Task Title:** Understanding and Implementation of Vector Databases

Extend the clinical knowledgebase by generating vector embeddings, storing them in vector indexes, and retrieving semantically related clinical passages.

### Key Deliverables

- Sentence Transformer embedding generation.
- Vector normalization.
- ChromaDB indexing.
- FAISS indexing.
- Query embedding generation.
- Top-K similarity search.
- Source-metadata retrieval.
- Browser-based search interface.

## 9. Understanding Vector Embeddings

Vector embeddings are numerical representations of text created by an embedding model.

The model maps related textual concepts into a numerical space where similarity can be measured mathematically.

### General-Purpose Model

**Model ID:** `sentence-transformers/all-MiniLM-L6-v2`

**Embedding dimensions:** 384

For 631 clinical chunks, the expected embedding matrix shape is `(631, 384)`.

This represents 631 chunk embeddings, each containing 384 numerical dimensions.

### Embedding Normalization

L2 normalization scales nonzero vectors to unit length.

When both query and document embeddings are normalized, their dot product equals cosine similarity.

The same embedding model should be used for indexing documents and encoding search queries.

## 10. ChromaDB Implementation

ChromaDB is a vector database supporting:

- Vector storage
- Document collections
- Metadata
- Similarity queries
- Local persistence
- Document retrieval

### Workflow

**Clinical Chunks → Embedding Model → ChromaDB Collection → Query Vector → Matching Documents**

The implementation stores document text and source metadata alongside embeddings.

## 11. FAISS Implementation

FAISS is a vector similarity search library developed by Meta AI.

The clinical retrieval implementation uses an exact inner-product index for normalized embeddings.

### Workflow

**Clinical Chunks → 384-Dimensional Embeddings → FAISS Index → Query Search → Matching Chunk IDs → Source Passages**

Unlike ChromaDB, FAISS requires the application to manage original text and metadata separately.

## 12. ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Type | Vector database | Vector search library |
| Vector indexing | Supported | Supported |
| Original text storage | Built in | Managed separately |
| Metadata storage | Built in | Managed separately |
| Similarity search | Supported | Supported |
| Local persistence | Supported | Index serialization |
| Primary use | Document retrieval | Efficient vector similarity search |

## 13. Semantic Search Workflow

The browser interface supports selecting a prepared chunk dataset, generating embeddings, building vector indexes, and searching clinical passages.

Example search query:

**"How should blood pressure be monitored?"**

The system retrieves related chunks according to embedding similarity.

Retrieved passages must be checked against their original source documents.

### Day 2 Status

Implementation has been prepared and integrated into the main Streamlit application.

Full browser-based indexing, persistence, and retrieval relevance testing should be independently confirmed before claiming production readiness.

---

# DAY 3 — GENERAL-PURPOSE VS BIOMEDICAL EMBEDDING EVALUATION

## 14. Task Description

**Task Title:** Evaluation of General-Purpose and Domain-Oriented Embedding Models

Compare two embedding models using medical sentence pairs covering synonyms, abbreviations, negation, and numerical differences.

Generate embeddings, calculate cosine similarities, visualize results with PCA, and identify strengths and limitations.

### Key Deliverables

- Two embedding models.
- A dataset of at least 25 medical sentence pairs.
- Cosine similarity calculations.
- Category-wise performance comparison.
- PCA visualizations.
- Evaluation CSV files.
- Streamlit evaluation dashboard.
- Research findings and documentation.

## 15. Selected Models

| Property | General Model | Biomedical Model |
|---|---|---|
| Model | MiniLM | BiomedBERT embeddings |
| Model ID | `sentence-transformers/all-MiniLM-L6-v2` | `NeuML/biomedbert-base-embeddings` |
| Dimensions | 384 | 768 |
| Focus | General language | Biomedical language |

Embedding dimensionality alone does not establish model quality.

## 16. Evaluation Dataset

A controlled dataset of 32 medical sentence pairs was prepared.

| Category | Number of Pairs |
|---|---:|
| Synonyms | 8 |
| Abbreviations | 8 |
| Negation | 8 |
| Numerical differences | 8 |
| Total | 32 |

Each pair contains two sentences, a category, a unique identifier, and an expected equivalence label.

**Label 1:** Equivalent or closely equivalent meaning.

**Label 0:** Clinically different statement.

The labels are simplified for exploratory evaluation.

### Example Pairs

**Synonyms:** "The patient has hypertension." / "The patient has high blood pressure."

**Abbreviations:** "The patient has COPD." / "The patient has chronic obstructive pulmonary disease."

**Negation:** "The patient has pneumonia." / "The patient does not have pneumonia."

**Numerical difference:** "The patient's potassium level is 4.0 mmol/L." / "The patient's potassium level is 6.5 mmol/L."

## 17. Cosine Similarity

Cosine similarity measures the directional alignment between two vectors.

**Formula:** `cosine(A,B) = (A · B) / (||A|| × ||B||)`

Higher cosine similarity indicates stronger vector alignment.

However, high similarity does not necessarily imply clinical equivalence or logical agreement.

## 18. Day 3 Experimental Results

Both models executed successfully.

| Model | Output Shape |
|---|---|
| General MiniLM | (64, 384) |
| Biomedical BiomedBERT | (64, 768) |

### Category-Wise Mean Cosine Similarity

| Category | General | Biomedical |
|---|---:|---:|
| Synonyms | 0.7585 | 0.8672 |
| Abbreviations | 0.6389 | 0.8797 |
| Negation | 0.8132 | 0.9773 |
| Numerical differences | 0.9155 | 0.9957 |

### Interpretation

**Synonyms:** The biomedical model produced higher average similarity for equivalent medical terminology.

**Abbreviations:** The biomedical model recognized most abbreviation expansions more strongly, although some exceptions occurred.

**Negation:** Both models assigned high similarity to contradictory medical statements, with the biomedical model producing especially high scores.

**Numerical differences:** Both models assigned high similarity to sentences containing different clinical measurements.

These results demonstrate the limitations of using embedding similarity alone to determine clinical equivalence.

## 19. Important Model Findings

### Abbreviation Example — A07

"The patient has SOB."

"The patient has shortness of breath."

| Model | Similarity |
|---|---:|
| General | 0.7525 |
| Biomedical | 0.3149 |

This result highlights a possible abbreviation ambiguity or representation weakness requiring further investigation.

### Negation Example — N01

"The patient has pneumonia."

"The patient does not have pneumonia."

| Model | Similarity |
|---|---:|
| General | 0.8435 |
| Biomedical | 0.9781 |

The biomedical model represented contradictory statements as highly similar.

### Numerical Example — D02

"The patient's potassium level is 4.0 mmol/L."

"The patient's potassium level is 6.5 mmol/L."

| Model | Similarity |
|---|---:|
| General | 0.9518 |
| Biomedical | 0.9997 |

This illustrates why exact numerical validation may be necessary in clinical retrieval.

## 20. Exploratory Discrimination

| Metric | General | Biomedical |
|---|---:|---:|
| Equivalent mean | 0.6987 | 0.8735 |
| Different mean | 0.8644 | 0.9865 |
| Mean separation | -0.1656 | -0.1130 |

The reported biomedical exploratory ROC-AUC was 0.0625.

The negative separation values indicate that the clinically different pairs received higher average similarity than the equivalent pairs in this benchmark.

An important limitation is that negative examples often have very similar wording, while positive examples use paraphrases.

Therefore, this experiment should not be interpreted as a definitive clinical model ranking.

## 21. PCA Visualization

Principal Component Analysis reduces high-dimensional embeddings into two dimensions for visualization.

Generated files:

- `evaluation_results/pca_general.png`
- `evaluation_results/pca_biomedical.png`

### Visualization Categories

- Green: Synonyms
- Blue: Abbreviations
- Red: Negation
- Orange: Numerical differences

Circle markers represent the first sentence in a pair, while cross markers represent the second.

The PCA projections were fitted separately for the two models. Their plotted coordinates are not directly comparable.

## 22. Day 3 Evaluation Dashboard

**Streamlit page:** `pages/3_Embedding_Model_Evaluation.py`

Features include:

- Experiment overview.
- Model dimensions.
- Category-wise comparison.
- Individual sentence-pair explorer.
- Cosine similarity charts.
- Exploratory discrimination metrics.
- PCA visualizations.
- CSV and report downloads.
- Research interpretation.

The dashboard uses the predefined sentence-pair experiment and does not require uploaded clinical PDFs.

### Day 3 Status

The 32-pair CLI experiment ran successfully, and evaluation results and PCA images were generated.

The feature branch `feature/embedding-model-evaluation` was pushed to GitHub.

Complete browser testing remains to be confirmed.

---

# DAY 4 — SIMILARITY METRICS & VECTOR SEARCH PERFORMANCE

## 23. Task Description

**Task Title:** Performance Evaluation of Similarity Metrics and Vector Search Algorithms

Implement and compare cosine similarity, dot product, and Euclidean distance using NumPy.

Build a Top-K nearest-neighbor search system and compare its performance with FAISS exact and approximate search.

Evaluate at least 50,000 vectors, measure search latency and accuracy, and visualize search time against dataset size and the approximate-search speed–recall trade-off.

### Key Deliverables

- NumPy cosine similarity.
- NumPy dot product.
- NumPy Euclidean distance.
- Top-K vector search.
- FAISS exact search.
- FAISS approximate search.
- A dataset containing at least 50,000 vectors.
- Search time measurements.
- Recall@K evaluation.
- Performance visualizations.
- Benchmark CSV files.
- Research findings.

## 24. Similarity Metrics

### 24.1 Cosine Similarity

Measures directional similarity between two nonzero vectors.

**Formula:** `cosine(A,B) = (A · B) / (||A|| × ||B||)`

Higher scores indicate greater directional alignment.

### 24.2 Dot Product

Measures the inner product between vectors.

**Formula:** `A · B = Σ(Ai × Bi)`

Raw dot product is influenced by vector magnitude.

For normalized vectors, dot product equals cosine similarity.

### 24.3 Euclidean Distance

Measures geometric distance between vectors.

**Formula:** `d(A,B) = sqrt(Σ(Ai - Bi)²)`

Lower distance means closer vectors.

The benchmark calculates squared Euclidean distance for ranking, which produces the same ordering as Euclidean distance.

## 25. Search Algorithms

### 25.1 NumPy Exhaustive Search

Implemented vectorized exact search using NumPy.

Each query is compared against every dataset vector, and the Top-K results are selected.

### 25.2 FAISS Exact Search

Used `faiss.IndexFlatIP` with L2-normalized vectors.

This performs exhaustive inner-product nearest-neighbor search.

It was compared against normalized NumPy cosine retrieval.

### 25.3 FAISS Approximate Search

Used `faiss.IndexIVFFlat`.

This algorithm groups vectors into inverted lists and searches selected lists for each query.

**nlist:** Number of inverted lists.

**nprobe:** Number of lists searched for a query.

Increasing nprobe generally improves nearest-neighbor recall while increasing search computation.

## 26. Experimental Configuration

| Parameter | Configuration |
|---|---|
| Dataset | Synthetic clustered vectors |
| Dataset sizes | 10,000 / 25,000 / 50,000 |
| Dimensions | 128 |
| Queries per size | 32 |
| Top-K | 10 |
| Timing repeats | 5 |
| Timing statistic | Median after warm-up |
| Random seed | 42 |
| FAISS exact index | IndexFlatIP |
| FAISS approximate index | IndexIVFFlat |
| IVF nprobe | 1, 4, 16, 32 |
| IVF nlist | 100, 158, 223 |
| IVF training examples | 10,000 or 12,000 |

The benchmark uses synthetic embeddings instead of real clinical text embeddings.

## 27. Recorded Benchmark Results

All timing results were obtained from a local CLI execution.

**Search latency represents the median time for a batch of 32 queries.**

### Search Time vs Dataset Size

| Search Method | 10k Vectors | 25k Vectors | 50k Vectors |
|---|---:|---:|---:|
| NumPy Cosine | 12.939 ms | 31.798 ms | 68.471 ms |
| NumPy Dot | 11.812 ms | 30.846 ms | 65.222 ms |
| NumPy Euclidean | 21.711 ms | 55.210 ms | 124.808 ms |
| FAISS Exact | 119.073 ms | 186.722 ms | 376.940 ms |
| IVF nprobe=1 | 1.187 ms | 2.232 ms | 3.877 ms |
| IVF nprobe=4 | 4.214 ms | 6.065 ms | 9.214 ms |
| IVF nprobe=16 | 13.391 ms | 21.252 ms | 30.054 ms |
| IVF nprobe=32 | 26.527 ms | 41.268 ms | 58.213 ms |

### Approximate Recall@10

| IVF Configuration | 10k | 25k | 50k |
|---|---:|---:|---:|
| nprobe=1 | 0.9406 | 0.8625 | 0.7344 |
| nprobe=4 | 1.0000 | 1.0000 | 1.0000 |
| nprobe=16 | 1.0000 | 1.0000 | 1.0000 |
| nprobe=32 | 1.0000 | 1.0000 | 1.0000 |

### Exact Search Verification

FAISS exact search matched normalized NumPy cosine Top-K results on all tested dataset sizes.

**Observed Recall@10 agreement: 1.0000**

This verifies agreement between the exact implementations for the tested workload.

## 28. Key Day 4 Findings

### Finding 1 — Fastest Approximate Search

At 50,000 vectors, FAISS IVF with nprobe=1 achieved:

- Batch search time: **3.877 ms**
- Recall@10: **73.44%**

It was the fastest tested configuration but missed some exact nearest neighbors.

### Finding 2 — Best Observed Speed–Recall Trade-Off

At 50,000 vectors, FAISS IVF with nprobe=4 achieved:

- Batch search time: **9.214 ms**
- Recall@10: **100%**

This provided the best observed speed–recall balance among the tested settings on this particular synthetic dataset.

It does not guarantee 100% recall on other datasets.

### Finding 3 — nprobe Effect

Increasing nprobe improved retrieval agreement with exact search while increasing latency.

### Finding 4 — NumPy vs FAISS Exact

At 50,000 vectors:

- NumPy cosine: **68.471 ms**
- FAISS exact: **376.940 ms**

FAISS exact was slower in this specific run.

The benchmark explicitly configured FAISS to use one thread, while NumPy BLAS threading was not controlled in the same way.

Therefore, this result is not evidence that FAISS exact search is universally slower than NumPy.

### Finding 5 — Index Building vs Query Time

Approximate indexes have training and index-building costs.

At 50,000 vectors, the recorded IVF build/training time was approximately **625.427 ms**.

Query-time comparisons should not be confused with total initialization cost.

## 29. Recall@K Evaluation

Recall@K measures how many exact nearest neighbors are recovered by approximate search.

**Formula:** `Recall@K = Number of matching exact neighbors / K`

For example, recovering eight exact neighbors out of ten gives Recall@10 = 0.8.

Recall@K in this benchmark measures nearest-neighbor agreement, not clinical relevance or medical correctness.

## 30. Day 4 Generated Files

- `benchmarks/benchmark.py`
- `benchmark_results/benchmark_results.csv`
- `benchmark_results/accuracy_results.csv`
- `benchmark_results/search_time_vs_dataset_size.png`
- `benchmark_results/ivf_speed_recall_tradeoff.png`

### Graph 1 — Search Time vs Dataset Size

Displays how median search latency changes between 10,000, 25,000, and 50,000 vectors for the tested search methods.

### Graph 2 — FAISS IVF Speed–Recall Trade-Off

Displays the relationship between approximate-search query time and Recall@10 at 50,000 vectors.

## 31. Day 4 Browser Dashboard

**Page:** `pages/4_Vector_Search_Performance.py`

The dashboard integration is designed for:

- Viewing benchmark results.
- Comparing search methods.
- Displaying performance graphs.
- Exploring approximate-search recall.
- Testing vector retrieval interactively.
- Downloading research outputs.
- Checking benchmark prerequisites.

The CLI benchmark has been executed successfully.

Complete browser functionality must still be verified on the local machine.

---

# APPLICATION SETUP AND USAGE

## 32. Project Structure

The repository is organized around the following files and folders:

| Path | Purpose |
|---|---|
| `app.py` | Main Streamlit application, Day 1 and Day 2 |
| `build_knowledgebase.py` | Clinical knowledgebase CLI |
| `kb/` | Document processing and validation |
| `vector_store/` | Vector embedding utilities |
| `embedding_evaluation/` | Day 3 model comparison scripts |
| `benchmarks/benchmark.py` | Day 4 performance benchmark |
| `pages/3_Embedding_Model_Evaluation.py` | Day 3 dashboard |
| `pages/4_Vector_Search_Performance.py` | Day 4 dashboard |
| `pages/1_Guided_Clinical_Workflow.py` | Optional guided processing page |
| `services/clinical_workflow.py` | Optional guided workflow backend |
| `output/` | Generated clinical knowledgebase artifacts |
| `evaluation_results/` | Day 3 results and PCA graphs |
| `benchmark_results/` | Day 4 benchmark results |
| `vector_data/` | Local vector indexes |
| `README.md` | Project documentation |

Guided workflow files are available only if copied into the local project.

The exact directory contents may vary depending on which modules and integrations are installed.

## 33. Installation

### Step 1 — Clone Repository

`git clone https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation.git`

`cd clinical-knowledgebase-preparation`

### Step 2 — Create a Virtual Environment

On Windows PowerShell:

`python -m venv .venv`

Activate it:

`.\.venv\Scripts\Activate.ps1`

### Step 3 — Install Dependencies

`python -m pip install -r requirements.txt`

Ensure the full workflow dependencies are installed:

`python -m pip install streamlit sentence-transformers chromadb faiss-cpu numpy pandas scikit-learn matplotlib`

The clinical document processing modules also require the extraction dependencies listed in the project requirements.

### Step 4 — Hugging Face Cache Configuration

On Windows, Hugging Face may download model files into the C: drive by default.

If storage space is limited, configure another drive before launching the Python process.

PowerShell:

`$env:HF_HOME = "D:\huggingface_cache"`

`$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"`

`$env:HF_XET_CACHE = "D:\huggingface_cache\xet"`

These environment variables apply to the current PowerShell session.

## 34. Run the Streamlit Application

From the repository root:

`python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501`

Open:

http://127.0.0.1:8501

The application uses Streamlit multipage navigation when additional pages are installed in the `pages/` directory.

### Windows Environment Note

The local setup used:

- Python 3.14.4
- Streamlit 1.67.0
- Uvicorn 0.54.0

The Streamlit server was able to start, but repeated Windows asyncio `WinError 10054` connection-reset messages were observed.

These messages should be investigated if the application disconnects or fails to render. Successful server startup alone does not prove that all browser workflows function correctly.

## 35. Run Day 1 Through CLI

`python build_knowledgebase.py`

This executes the clinical knowledgebase processing pipeline using the configured input documents.

## 36. Run Day 3 Evaluation

Initial comparison:

`python -m embedding_evaluation.compare_models`

Full evaluation:

`python -m embedding_evaluation.evaluate`

Generate research report:

`python -m embedding_evaluation.report`

## 37. Run Day 4 Benchmark

`python benchmarks/benchmark.py`

This generates the benchmark timing results, Recall@10 values, CSV files, and performance graphs.

---

# BROWSER TESTING GUIDE

## 38. Day 1 — Document Upload

1. Start Streamlit.
2. Open the Clinical Knowledgebase Builder or installed Guided Workflow page.
3. Upload public or synthetic clinical documents.
4. Click the knowledgebase processing button.
5. Review extracted text and section statistics.
6. Inspect generated chunks and source metadata.
7. Download the generated JSONL and CSV artifacts.

Different source documents will produce different chunk counts.

## 39. Day 2 — Semantic Search

1. Use the prepared chunk dataset from Day 1.
2. Select the appropriate chunk source.
3. Generate embeddings.
4. Build ChromaDB and FAISS indexes.
5. Enter a search query relevant to the documents.
6. Set Top-K to 3.
7. Compare ChromaDB and FAISS results.
8. Check returned passages against the original document.

Example query:

**"How should blood pressure be monitored?"**

## 40. Day 3 — Embedding Evaluation

Open the Embedding Model Evaluation page.

Check:

- 32 sentence pairs.
- 64 sentence instances.
- 384-dimensional general embeddings.
- 768-dimensional biomedical embeddings.
- Four evaluation categories.
- Pair-wise cosine scores.
- Two PCA graphs.

Example recorded result for synonym pair S01:

General = 0.8828; Biomedical = 0.9511.

## 41. Day 4 — Vector Search Benchmark

Open the Vector Search Performance page if installed.

Check:

- Dataset sizes 10k, 25k, and 50k.
- NumPy and FAISS methods.
- Search-time comparison chart.
- Approximate Recall@10.
- IVF nprobe settings.
- Speed–recall trade-off graph.
- Result CSV downloads.

Expected recorded result at 50,000 vectors:

**IVF nprobe=4: 9.214 ms batch time and 100% Recall@10.**

These values describe the previous benchmark execution. New runs may produce different timing results.

## 42. Guided End-to-End Workflow

The optional guided workflow is intended to simplify the process for users.

Its workflow is:

**Upload → Extract → Chunk → Validate → Generate Embeddings → Build Indexes → Semantic Search**

Day 3 and Day 4 remain independent evaluation modules using their own datasets.

Uploading a PDF does not automatically create the 32 labeled model-evaluation pairs or a 50,000-vector benchmark.

---

# DEVELOPMENT, TESTING & LIMITATIONS

## 43. Development Status

| Module | Implementation | Verified Status |
|---|---|---|
| Day 1 Document Preparation | Completed | CLI executed successfully |
| Day 2 Vector Search | Implemented | Full browser testing pending confirmation |
| Day 3 Model Evaluation | Completed | 32-pair CLI experiment successful |
| Day 3 PCA | Generated | Output files created |
| Day 3 Dashboard | Implemented | Browser testing pending confirmation |
| Day 4 NumPy Metrics | Completed | CLI benchmark successful |
| Day 4 FAISS Exact | Completed | CLI benchmark successful |
| Day 4 FAISS Approximate | Completed | CLI benchmark successful |
| Day 4 50k Benchmark | Completed | Actual numerical results obtained |
| Day 4 CSV and Charts | Generated | Local output reported |
| Day 4 Dashboard | Integration prepared | Browser verification pending |
| Guided Workflow | Integration prepared | End-to-end verification pending |

## 44. Current Limitations

### Document Processing

- Complex PDF layouts can affect extraction.
- Scanned documents depend on OCR quality.
- Tables may lose some formatting.
- Section detection uses heuristics.
- Chunk lengths may not align perfectly with embedding-model token limits.

### Semantic Retrieval

- High cosine similarity does not guarantee correct clinical meaning.
- Negation and numerical changes may be insufficiently distinguished.
- Retrieval performance depends on document quality and embedding choice.
- Labeled relevance datasets are needed to measure real clinical retrieval quality.

### Embedding Evaluation

- The medical benchmark contains only 32 pairs.
- Equivalence labels are simplified.
- Some medical abbreviations are ambiguous.
- The benchmark's lexical-overlap patterns may influence performance.
- PCA plots are descriptive, not direct accuracy measurements.

### Vector Search Performance

- Day 4 uses synthetic 128-dimensional vectors.
- Search performance depends on hardware and dataset distribution.
- FAISS and NumPy CPU threading were not fully matched.
- IVF training and index-building time must be considered separately.
- Approximate recall against exact search does not establish medical relevance.
- Additional random seeds and realistic datasets would improve the evaluation.

### Clinical Safety

This application is not validated for medical diagnosis, treatment decisions, or autonomous clinical recommendations.

---

## 45. Future Improvements

- Complete end-to-end Streamlit browser testing.
- Connect uploaded documents to a guided retrieval evaluation workflow.
- Benchmark real 384-dimensional clinical embeddings.
- Add labeled document–query relevance datasets.
- Calculate Recall@K, Precision@K, and MRR for clinical retrieval.
- Implement controlled multithreading benchmarks.
- Test HNSW and additional approximate indexes.
- Evaluate larger datasets.
- Improve negation and numerical handling.
- Add hybrid lexical and semantic search.
- Explore reranking and source-grounded RAG.
- Introduce automated tests and regression checks.

---

# GITHUB & PROJECT MANAGEMENT

## 46. GitHub Repository

https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation

### Development Branches

- Day 2: `feature/vector-databases`
- Day 3: `feature/embedding-model-evaluation`
- Day 4: `feature/vector-search-performance` (use if created locally)

Day 3 was successfully pushed to GitHub.

Day 4 push status should be checked using Git before claiming it is published.

## 47. GitHub Workflow

First check the current branch and repository status:

`git status`

`git branch --show-current`

Review changes before staging.

Common Day 4 files:

- `README.md`
- `benchmarks/benchmark.py`
- `benchmark_results/benchmark_results.csv`
- `benchmark_results/accuracy_results.csv`
- `benchmark_results/search_time_vs_dataset_size.png`
- `benchmark_results/ivf_speed_recall_tradeoff.png`

Only stage optional Streamlit or guided workflow files if they are installed and reviewed.

Commit changes and push the active feature branch. Create a pull request after successful testing.

## 48. Data Privacy and Repository Hygiene

Do not commit:

- Private patient documents.
- Patient-identifiable information.
- Sensitive extracted clinical text.
- Secrets or API keys.
- Virtual environments.
- Local model caches.
- Local vector indexes.
- Python cache files.

Recommended ignore patterns:

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `.env`
- `.env.*`
- `input_docs/`
- `output/`
- `vector_data/`
- `huggingface_cache/`

Synthetic benchmark CSVs and graphs may be committed when they contain no private information.

---

# INTERNSHIP CONTRIBUTION & AI ASSISTANCE

## 49. My Contribution

I worked on the Clinical AI Research Workbench through four connected AI/ML engineering tasks.

For Day 1, I executed and validated the clinical document-processing pipeline using two guidelines and reviewed the generated sections and text chunks.

For Day 2, I worked with embedding generation, vector indexing, and semantic retrieval using ChromaDB and FAISS.

For Day 3, I executed both embedding models, evaluated 32 medical sentence pairs, examined cosine similarities, and generated PCA visualizations.

For Day 4, I executed NumPy and FAISS benchmarks using 10,000, 25,000, and 50,000 vectors, reviewed search latency and Recall@10, and generated performance comparison graphs.

I also worked on organizing the project, integrating research modules into Streamlit, reviewing results, and maintaining GitHub documentation.



## 51. Tools Used

- Visual Studio Code
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Sentence Transformers
- Hugging Face
- ChromaDB
- FAISS
- Matplotlib
- PyMuPDF
- Tesseract OCR
- Git
- GitHub
- ChatGPT

## 52. AI Usage Summary

- **Human work and decisions:** 60%
- **AI assistance:** 40%

These are self-assessed contribution estimates, not independently measured percentages.

---

# FINAL RESEARCH CONCLUSION

## 53. Project Outcomes

The project demonstrates four connected aspects of AI retrieval engineering.

**Day 1:** Clinical documents can be transformed into structured, metadata-rich chunks suitable for vector retrieval.

**Day 2:** Numerical embeddings and vector indexes allow semantic retrieval of related text passages.

**Day 3:** Specialized biomedical embeddings can represent medical terminology effectively but may still struggle with negation, numerical differences, and ambiguous abbreviations.

**Day 4:** Approximate nearest-neighbor search can substantially reduce retrieval latency while introducing a measurable trade-off in nearest-neighbor recall.

The Day 4 benchmark identified FAISS IVF with nprobe=4 as the best observed speed–recall configuration among the tested settings at 50,000 synthetic vectors, with a median 32-query batch time of 9.214 ms and Recall@10 of 1.0000.

These findings are exploratory, hardware-dependent, and dataset-specific.

Together, the four tasks establish a practical foundation for further research into scalable semantic retrieval, biomedical embeddings, vector databases, and RAG systems.

