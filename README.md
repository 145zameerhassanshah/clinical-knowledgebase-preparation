
# Clinical Knowledgebase & Biomedical Embedding Evaluation

### End-to-End Clinical Document Processing, Vector Search & Embedding Model Evaluation

A Python and Streamlit-based clinical knowledge engineering project that transforms medical documents into structured, retrieval-ready datasets, enables semantic search using ChromaDB and FAISS, and evaluates general-purpose versus biomedical embedding models.

The project was developed as part of practical AI/ML, vector database, and Retrieval-Augmented Generation (RAG) engineering learning.

---

## 1. Project Overview

Clinical guidelines, medical protocols, and healthcare documents often contain complex terminology, abbreviations, measurements, and specialized language.

Building a reliable clinical AI retrieval system involves three important stages:

**Day 1 — Clinical Knowledgebase Preparation**

Convert raw clinical documents into structured, metadata-rich text chunks.

**Day 2 — Vector Databases & Semantic Search**

Convert clinical chunks into numerical embeddings and retrieve relevant passages using ChromaDB and FAISS.

**Day 3 — Embedding Model Evaluation**

Compare general-purpose and biomedical embedding models using medical sentence pairs, cosine similarity, category-wise evaluation, and PCA visualization.

Together, these stages form a foundation for developing and evaluating clinical RAG systems.

---

## 2. Real-World Problem

Healthcare organizations maintain large collections of:

- Clinical practice guidelines
- Medical research documents
- Hospital protocols
- Standard operating procedures
- Patient management guidelines
- Treatment and monitoring recommendations

Traditional keyword search may fail when medical terms have different wording.

Example:

Document:

"The patient presents with dyspnea."

Query:

"The patient has shortness of breath."

Both statements describe a related clinical concept.

However, medical retrieval systems must also distinguish clinically important differences.

Example:

"The patient has pneumonia."

"The patient does not have pneumonia."

The sentences share almost the same words but have contradictory meanings.

Similarly:

"The patient's potassium level is 4.0 mmol/L."

"The patient's potassium level is 6.5 mmol/L."

The statements describe the same measurement but different values.

### Problem Being Addressed

A clinical semantic retrieval system must recognize medical terminology while preserving important differences involving:

- Synonyms
- Medical abbreviations
- Negation
- Numerical measurements
- Medication doses
- Clinical context

The project investigates how different embedding models represent these relationships.

This is a research prototype and is not intended for clinical decision-making.

---

## 3. Complete System Architecture

```text
Clinical Documents
        |
        v
Document Upload
        |
        v
File Type Detection
        |
        v
Text Extraction / OCR
        |
        v
Logical Section Detection
        |
        v
Structure-Aware Chunking
        |
        v
Metadata Preservation
        |
        v
Validation
        |
        v
Retrieval-Ready Dataset
        |
        v
output/chunks.jsonl
        |
        v
General Embedding Model
all-MiniLM-L6-v2
        |
        v
384-Dimensional Embeddings
        |
        +------------------+
        |                  |
        v                  v
     ChromaDB             FAISS
        |                  |
        +---------+--------+
                  |
                  v
          Semantic Search
                  |
                  v
        Top-K Relevant Chunks
                  |
                  v
       Source Metadata + Scores


DAY 3: MODEL EVALUATION

32 Medical Sentence Pairs
        |
        +------------------------+
        |                        |
        v                        v
General Embedding Model    Biomedical Model
all-MiniLM-L6-v2           BiomedBERT
384 dimensions             768 dimensions
        |                        |
        +------------+-----------+
                     |
                     v
              Cosine Similarity
                     |
                     v
             Category Analysis
                     |
                     v
             PCA Visualization
                     |
                     v
             Model Comparison
                     |
                     v
          Evaluation Dashboard
                     |
                     v
        Final Research Findings
```

---

# DAY 1 — CLINICAL KNOWLEDGEBASE PREPARATION

## 4. Multi-Format Document Processing

The system supports:

- PDF
- DOCX
- TXT
- HTML / HTM
- PNG
- JPG / JPEG
- TIFF
- BMP
- WEBP

### Technologies

- PyMuPDF for PDF extraction
- python-docx for Word documents
- BeautifulSoup for HTML parsing
- Tesseract for OCR
- Python for processing
- Pandas for data inspection
- Streamlit for browser interaction

---

## 5. Document Processing Pipeline

### 5.1 Text Extraction

Extracts text from digital and scanned clinical documents.

### 5.2 Logical Section Detection

Identifies document sections such as:

- Assessment
- Diagnosis
- Treatment
- Monitoring
- Follow-up
- Contraindications
- Emergency Management

### 5.3 Structure-Aware Chunking

Divides extracted documents into retrieval-sized chunks while attempting to preserve sentence and section boundaries.

```text
Document
   |
   v
Section
   |
   v
Sentences
   |
   v
Text Chunks
```

### 5.4 Metadata Preservation

Each chunk retains source information.

Example:

```json
{
  "chunk_id": "guideline_s003_c002",
  "document_id": "guideline",
  "section_id": "guideline_s003",
  "section_title": "Monitoring",
  "chunk_index": 2,
  "text": "Clinical content...",
  "source_file": "guideline.pdf",
  "source_type": "pdf",
  "extraction_method": "PyMuPDF"
}
```

Metadata allows retrieved text to be traced back to its original document.

---

## 6. Day 1 Validation Results

The document processing pipeline was tested using two clinical guideline documents.

| Metric | Result |
|---|---:|
| Documents discovered | 2 |
| Documents processed | 2 |
| Logical sections | 168 |
| Retrieval chunks | 631 |
| Average chunk size | 720.64 characters |
| Failed documents | 0 |
| Retrieval preparation | READY |

### Generated Files

```text
output/
├── documents.json
├── sections.json
├── chunks.jsonl
├── chunks.csv
└── build_report.json
```

### Day 1 Outcome

Successfully transformed two clinical documents into 631 retrieval-ready text chunks with source metadata and validation information.

---

# DAY 2 — VECTOR DATABASES & SEMANTIC SEARCH

## 7. Understanding Vector Embeddings

Vector embeddings are numerical representations of text produced by an embedding model.

Semantically related text often produces vectors with similar directions.

Example:

```text
"The patient has hypertension."
              |
              v
      Embedding Model
              |
              v
   [0.12, -0.34, 0.87, ...]
```

The numeric values shown are illustrative.

### General-Purpose Embedding Model

Model:

`sentence-transformers/all-MiniLM-L6-v2`

Embedding dimensions: 384

For 631 text chunks, the expected embedding matrix shape is:

```text
(631, 384)
```

Each row represents one clinical text chunk.

Each column represents one embedding dimension.

---

## 8. ChromaDB Implementation

ChromaDB is a vector database that supports:

- Vector storage
- Document collections
- Metadata
- Similarity search
- Local persistence
- Document retrieval

### Workflow

```text
Clinical Chunks
       |
       v
Text Embeddings
       |
       v
ChromaDB Collection
       |
       v
Semantic Query
       |
       v
Relevant Clinical Passages
```

The system is designed to maintain document text and source information alongside embeddings.

---

## 9. FAISS Implementation

FAISS is a vector similarity search library developed by Meta AI.

It supports efficient nearest-neighbor retrieval and multiple vector indexing approaches.

### Workflow

```text
Clinical Chunks
       |
       v
384-Dimensional Vectors
       |
       v
FAISS Index
       |
       v
Query Embedding
       |
       v
Nearest-Neighbor Search
       |
       v
Matching Chunk IDs
       |
       v
Original Documents
```

Original text and metadata are managed separately by the application.

---

## 10. ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Type | Vector database | Vector search library |
| Vector indexing | Yes | Yes |
| Original text storage | Built in | External |
| Metadata management | Built in | External |
| Persistence | Persistent collections | Saved indexes |
| Similarity search | Yes | Yes |
| Primary purpose | Document retrieval | Vector similarity indexing |

### Day 2 Learning Outcome

Learned how vector embeddings can be indexed and retrieved through ChromaDB and FAISS.

The implementation supports semantic search through the Streamlit application.

Full browser integration and retrieval accuracy should be verified independently before claiming production readiness.

---

# DAY 3 — GENERAL-PURPOSE VS BIOMEDICAL EMBEDDING EVALUATION

## 11. Evaluation Objective

The objective is to compare two embedding models and understand how effectively they represent medical sentences.

The evaluation focuses on:

1. Medical synonyms
2. Medical abbreviations
3. Negation
4. Numerical differences

Each model converts the same medical sentences into numerical vectors.

Cosine similarity is then calculated to measure embedding alignment.

PCA is used to visualize the resulting high-dimensional embeddings.

---

## 12. Selected Embedding Models

### Model A — General-Purpose

Model ID:

`sentence-transformers/all-MiniLM-L6-v2`

Dimensions: 384

Purpose:

General semantic similarity, clustering, and text retrieval.

### Model B — Biomedical

Model ID:

`NeuML/biomedbert-base-embeddings`

Dimensions: 768

Purpose:

Biomedical-language embedding and retrieval tasks.

### Comparison

| Property | General | Biomedical |
|---|---|---|
| Model | MiniLM | BiomedBERT embeddings |
| Dimensions | 384 | 768 |
| Language focus | General | Biomedical |
| Sentence embeddings | Yes | Yes |
| Cosine similarity | Yes | Yes |
| PCA visualization | Yes | Yes |

The biomedical model has more dimensions, but dimensionality alone does not establish better performance.

---

## 13. Medical Sentence-Pair Dataset

A controlled evaluation dataset of 32 medical sentence pairs was created.

| Category | Pairs |
|---|---:|
| Synonyms | 8 |
| Abbreviations | 8 |
| Negation | 8 |
| Numerical differences | 8 |
| Total | 32 |

Each pair includes:

- Pair ID
- Category
- Sentence A
- Sentence B
- Expected equivalence label

### Ground Truth Labels

`1` = Equivalent or closely equivalent clinical meaning

`0` = Clinically different statement

These labels represent a simplified experimental benchmark.

### Synonym Example

Sentence A:

"The patient has hypertension."

Sentence B:

"The patient has high blood pressure."

Expected equivalence: 1

### Abbreviation Example

Sentence A:

"The patient has COPD."

Sentence B:

"The patient has chronic obstructive pulmonary disease."

Expected equivalence: 1

### Negation Example

Sentence A:

"The patient has pneumonia."

Sentence B:

"The patient does not have pneumonia."

Expected equivalence: 0

### Numerical Difference Example

Sentence A:

"The patient's potassium level is 4.0 mmol/L."

Sentence B:

"The patient's potassium level is 6.5 mmol/L."

Expected equivalence: 0

---

## 14. Cosine Similarity Evaluation

Cosine similarity compares the directions of two nonzero vectors.

Formula:

```text
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

For normalized vectors, cosine similarity equals their dot product.

Higher similarity means stronger directional alignment.

However:

High cosine similarity does not necessarily indicate:

- Clinical equivalence
- Logical entailment
- Absence of contradiction
- Identical numerical values
- Medical correctness

This distinction is particularly important for healthcare AI systems.

---

## 15. Day 3 Experiment Results

Both embedding models were executed successfully.

### Model Output

| Model | Embedding shape |
|---|---|
| General MiniLM | (64, 384) |
| Biomedical BiomedBERT | (64, 768) |

The 32 sentence pairs contain 64 sentence instances.

### Category-Wise Mean Cosine Similarity

| Category | General | Biomedical |
|---|---:|---:|
| Synonyms | 0.7585 | 0.8672 |
| Abbreviations | 0.6389 | 0.8797 |
| Negation | 0.8132 | 0.9773 |
| Numerical differences | 0.9155 | 0.9957 |

These are results from the executed experiment.

### Interpretation

**Synonyms**

Biomedical embeddings produced higher average similarity for equivalent medical terminology.

**Abbreviations**

Biomedical embeddings produced higher similarity for most abbreviation expansions.

**Negation**

Both models produced high similarity for clinically contradictory statements.

The biomedical model showed especially high similarity despite negation.

**Numerical differences**

Both models produced high similarity for statements containing different numerical measurements.

This suggests that cosine similarity alone is insufficient for detecting clinically important numerical changes.

---

## 16. Important Abbreviation Finding

Pair A07:

Sentence A:

"The patient has SOB."

Sentence B:

"The patient has shortness of breath."

### Results

| Model | Cosine similarity |
|---|---:|
| General | 0.7525 |
| Biomedical | 0.3149 |

Although the biomedical model generally performed well on abbreviations, it produced a low similarity score for this example.

This may indicate abbreviation ambiguity or representation limitations.

Further testing is required to identify the cause.

---

## 17. Negation and Numerical Findings

### Negation Example

Pair N01:

"The patient has pneumonia."

"The patient does not have pneumonia."

| Model | Cosine similarity |
|---|---:|
| General | 0.8435 |
| Biomedical | 0.9781 |

Both models represented the sentences as similar despite contradictory clinical meaning.

### Numerical Example

Pair D02:

"The patient's potassium level is 4.0 mmol/L."

"The patient's potassium level is 6.5 mmol/L."

| Model | Cosine similarity |
|---|---:|
| General | 0.9518 |
| Biomedical | 0.9997 |

These results illustrate the need for exact-value and negation-aware evaluation in medical retrieval systems.

---

## 18. Exploratory Equivalence Discrimination

The evaluation also compared equivalent and different sentence pairs.

### Recorded Results

| Metric | General | Biomedical |
|---|---:|---:|
| Equivalent mean | 0.6987 | 0.8735 |
| Different mean | 0.8644 | 0.9865 |
| Mean separation | -0.1656 | -0.1130 |

The biomedical model's exploratory ROC-AUC was reported as 0.0625.

A negative mean separation indicates that clinically different pairs received higher average similarity than equivalent pairs.

### Important Limitation

The benchmark uses closely worded negative pairs and more heavily paraphrased positive pairs.

Therefore, the results are influenced by lexical overlap.

The experiment is useful for studying embedding behavior but should not be considered a definitive measure of clinical retrieval performance.

---

## 19. PCA Visualization

Principal Component Analysis (PCA) was used to project sentence embeddings into two dimensions.

### General Model

Input shape:

```text
(64, 384)
```

Output:

```text
pca_general.png
```

### Biomedical Model

Input shape:

```text
(64, 768)
```

Output:

```text
pca_biomedical.png
```

### Visualization Categories

- Green: Synonyms
- Blue: Abbreviations
- Red: Negation
- Orange: Numerical differences

Circle markers represent first sentences.

Cross markers represent second sentences.

Connecting lines identify sentence pairs.

The two models were projected independently.

PCA graph positions are not directly comparable across models.

The primary quantitative comparison is based on original embedding cosine similarities.

---

## 20. Day 3 Streamlit Evaluation Dashboard

A separate Streamlit page was added:

```text
pages/3_Embedding_Model_Evaluation.py
```

The dashboard provides:

- Experiment overview
- Model dimensions
- Category-wise similarity comparison
- Interactive sentence-pair explorer
- Individual cosine scores
- Pair-wise comparison charts
- Equivalent vs different discrimination metrics
- PCA visualizations
- Real-world clinical AI interpretation
- Downloadable evaluation results

### Dashboard Metrics

- 32 sentence pairs
- 64 medical sentence instances
- 384 general embedding dimensions
- 768 biomedical embedding dimensions

The dashboard reads previously generated CSV files and PCA images.

It does not require a separate PDF upload for the controlled 32-pair experiment.

---

## 21. Updated Project Structure

```text
clinical-knowledgebase-preparation/
│
├── app.py
├── build_knowledgebase.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── kb/
│   ├── __init__.py
│   ├── models.py
│   ├── extractors.py
│   ├── sectioning.py
│   ├── chunking.py
│   ├── validation.py
│   └── builder.py
│
├── vector_store/
│   ├── __init__.py
│   └── embeddings.py
│
├── embedding_evaluation/
│   ├── __init__.py
│   ├── dataset.py
│   ├── compare_models.py
│   ├── evaluate.py
│   └── report.py
│
├── pages/
│   └── 3_Embedding_Model_Evaluation.py
│
├── input_docs/
│
├── output/
│   ├── documents.json
│   ├── sections.json
│   ├── chunks.jsonl
│   ├── chunks.csv
│   ├── build_report.json
│   └── embeddings.npy
│
├── vector_data/
│
└── evaluation_results/
    ├── similarity_scores.csv
    ├── category_summary.csv
    ├── pca_general.png
    ├── pca_biomedical.png
    ├── embeddings_general.npy
    ├── embeddings_biomedical.npy
    └── evaluation_report.md
```

Generated directories and files depend on which commands have been executed.

---

## 22. Installation

### Clone Repository

```bash
git clone https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation.git
```

```bash
cd clinical-knowledgebase-preparation
```

### Create Virtual Environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

For the complete three-stage workflow, ensure these packages are available:

```powershell
python -m pip install sentence-transformers chromadb faiss-cpu numpy pandas scikit-learn matplotlib
```

### Hugging Face Cache

On Windows, configure the model cache on a drive with sufficient free space:

```powershell
$env:HF_HOME = "D:\huggingface_cache"
$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"
$env:HF_XET_CACHE = "D:\huggingface_cache\xet"
```

These settings apply to the current PowerShell session.

---

## 23. Run the Complete Application

Start Streamlit:

```powershell
streamlit run app.py
```

Open:

http://localhost:8501

The project supports:

- Clinical document processing
- Knowledgebase generation
- Vector indexing and semantic search
- Embedding model evaluation

---

## 24. Complete Browser Testing Guide

### Test 1 — Clinical Document Processing

1. Open the Clinical Knowledgebase Builder.
2. Upload two public clinical guideline PDFs.
3. Click "Build Clinical Knowledgebase".
4. Review processing statistics.
5. Inspect generated sections and chunks.
6. Verify source metadata.
7. Download JSONL and CSV outputs.

Using the original two test documents, the previously recorded results were:

- 2 processed documents
- 168 sections
- 631 chunks
- 0 failed documents

Different documents may generate different results.

### Test 2 — ChromaDB and FAISS

1. Open the vector database section.
2. Select `Existing output/chunks.jsonl`.
3. Generate embeddings.
4. Build ChromaDB and FAISS indexes.
5. Verify the vector count.
6. Enter a query related to the documents.
7. Select comparison mode.
8. Set Top-K to 3.
9. Run semantic search.
10. Inspect returned chunks and source metadata.

Example:

"How should blood pressure be monitored?"

The results should be validated against the actual source documents.

### Test 3 — Embedding Model Evaluation

Open the "Embedding Model Evaluation" page.

Verify:

- 32 sentence pairs
- 64 sentence instances
- General model: 384 dimensions
- Biomedical model: 768 dimensions
- Four categories
- Category-wise charts
- Individual pair explorer
- PCA graphs
- Downloadable CSV results

### Test 4 — Synonym Example

Select:

Category: Synonyms

Pair: S01

Expected recorded similarity:

General: 0.8828

Biomedical: 0.9511

### Test 5 — Abbreviation Example

Select:

Category: Abbreviations

Pair: A07

Expected recorded similarity:

General: 0.7525

Biomedical: 0.3149

### Test 6 — Negation Example

Select:

Category: Negation

Pair: N01

Expected recorded similarity:

General: 0.8435

Biomedical: 0.9781

### Test 7 — Numerical Example

Select:

Category: Numerical

Pair: D02

Expected recorded similarity:

General: 0.9518

Biomedical: 0.9997

### Test 8 — PCA Visualization

Verify that:

- `pca_general.png` displays.
- `pca_biomedical.png` displays.
- Sentence-pair categories are identifiable.
- Both plots load without errors.

---

## 25. Run Evaluation Through CLI

### Initial Model Comparison

```powershell
python -m embedding_evaluation.compare_models
```

### Complete 32-Pair Evaluation

```powershell
python -m embedding_evaluation.evaluate
```

### Generate Research Report

```powershell
python -m embedding_evaluation.report
```

### Expected Generated Files

```text
evaluation_results/
├── similarity_scores.csv
├── category_summary.csv
├── pca_general.png
├── pca_biomedical.png
└── evaluation_report.md
```

---

## 26. GitHub Repository and Branches

Repository:

https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation

### Feature Branches

Day 2:

`feature/vector-databases`

Day 3:

`feature/embedding-model-evaluation`

The Day 3 feature branch has been pushed successfully.

### Push Further Updates

```powershell
git status
git branch --show-current
git add README.md
git commit -m "Document clinical embedding evaluation and browser testing"
git push
```

Review staged files before committing.

Create a pull request after testing is complete.

---

## 27. Safety and Data Privacy

The application is designed for research and educational use.

Do not upload private patient records unless appropriate authorization and safeguards are in place.

Avoid committing:

- Private clinical documents
- Patient information
- Private generated chunks
- API keys
- Model caches
- Virtual environments
- Local vector databases

Recommended `.gitignore` rules:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.*
input_docs/
output/
vector_data/
huggingface_cache/
```

Generated evaluation CSVs and visualizations may be committed when they contain only approved or non-sensitive synthetic research examples.

---

## 28. Current Limitations

### Document Preparation

- PDF structure detection uses heuristics.
- OCR quality depends on source scans.
- Complex medical tables may lose formatting.
- Character-based chunking may not align with embedding-model token limits.

### Vector Retrieval

- Semantic similarity does not guarantee clinical correctness.
- Negations and numerical values may be insufficiently distinguished.
- Retrieval quality depends on indexing and embedding configurations.
- Source alignment and metadata must be verified.

### Embedding Evaluation

- The benchmark includes only 32 sentence pairs.
- Ground-truth labels are simplified.
- Some abbreviations are ambiguous.
- Category-wise mean similarities do not independently establish accuracy.
- PCA is for visualization rather than direct accuracy measurement.
- Additional independent benchmarks are needed before model selection.

### RAG

LLM-based grounded answer generation has not been established as part of this completed evaluation.

The project is not a validated medical decision-support system.

---

## 29. Future Improvements

Potential extensions include:

1. Evaluation on real clinical document retrieval questions.
2. Manually labeled relevance datasets.
3. Recall@K, Precision@K, and MRR.
4. Improved biomedical abbreviation handling.
5. Negation-aware retrieval.
6. Exact numerical and unit validation.
7. Hybrid keyword and semantic retrieval.
8. Cross-encoder reranking.
9. Clinical-domain embedding benchmarks.
10. Source-grounded RAG generation.
11. Retrieval confidence calibration.
12. More extensive automated testing.

---

## 30. Key Learning Outcomes

### Day 1

- Multi-format document ingestion
- PDF extraction
- OCR
- Document section detection
- Structure-aware chunking
- Metadata management
- Retrieval-ready dataset preparation

### Day 2

- Text vector embeddings
- Embedding dimensions
- Vector normalization
- Cosine similarity
- ChromaDB
- FAISS
- Nearest-neighbor search
- Top-K retrieval
- Semantic document search

### Day 3

- General-purpose vs domain-oriented embeddings
- Biomedical language modeling
- Medical synonyms
- Clinical abbreviations
- Negation sensitivity
- Numerical differences
- Cosine similarity evaluation
- Exploratory ROC-AUC
- PCA visualization
- Model comparison
- Research result interpretation
- Streamlit evaluation dashboards

---

## 31. Development Status

| Stage | Implementation | Verification |
|---|---|---|
| Day 1 — Knowledgebase | Completed | CLI tested successfully |
| Day 2 — Vector Search | Implemented | Full browser verification should be confirmed |
| Day 3 — Model Evaluation | Completed | 32-pair CLI experiment executed |
| Day 3 — PCA | Generated | Browser display should be confirmed |
| Day 3 — Dashboard | Implemented | Browser testing should be confirmed |
| Day 3 — GitHub | Pushed | Feature branch available |

---

## 32. Final Research Conclusion

The experiment demonstrated that general-purpose and biomedical embedding models represent medical language differently.

The biomedical model produced higher average cosine similarity for medical synonyms and most abbreviation pairs.

However, it also assigned extremely high similarity to statements containing negation and clinically significant numerical changes.

The general-purpose model showed relatively greater separation for these clinically different pairs, although it also struggled to distinguish them adequately.

The results demonstrate that a specialized biomedical embedding model should not automatically be considered more reliable for all clinical retrieval tasks.

Model selection should account for terminology recognition, negation, numerical sensitivity, retrieval relevance, and practical safety requirements.

The project provides a reproducible foundation for evaluating embedding models before integrating them into more advanced clinical retrieval and RAG systems.

---

