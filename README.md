
# Clinical Knowledgebase Preparation & Vector Semantic Search

A multi-format clinical document processing and semantic retrieval system built with Python, Streamlit, Sentence Transformers, ChromaDB, and FAISS.

This project transforms clinical documents into structured, metadata-rich text chunks, generates vector embeddings, and enables semantic similarity search using two vector search technologies.

The project is developed as part of practical AI/ML, vector database, and Retrieval-Augmented Generation (RAG) engineering learning.

---

## 1. Project Overview

Clinical knowledge is commonly available in multiple formats:

- PDF clinical guidelines
- Word documents
- Text files
- HTML documents
- Scanned PDFs
- Medical document images

These documents require preprocessing before they can be used effectively in AI-powered retrieval systems.

The application provides an end-to-end workflow:

1. Document ingestion
2. Text extraction and OCR
3. Logical section detection
4. Structure-aware text chunking
5. Metadata and source preservation
6. Knowledgebase validation
7. Vector embedding generation
8. Vector indexing using ChromaDB and FAISS
9. Semantic similarity search
10. Retrieval result comparison

The system includes a browser-based Streamlit interface and a Python command-line workflow.

---

## 2. Problem Statement

Traditional keyword-based search may fail to retrieve relevant information when the user's query and the document use different wording.

For example:

Document:

"Patients with hypertension require regular blood pressure monitoring."

User query:

"How should high blood pressure be checked?"

Although the wording differs, both texts discuss related clinical concepts.

Vector embeddings allow the application to represent text numerically and search for semantically related information.

This improves the ability to retrieve relevant clinical passages without relying entirely on exact keyword matches.

Note: Semantic retrieval does not guarantee clinical accuracy. Retrieved results must be evaluated against their original sources.

---

## 3. System Architecture

```text
Clinical Documents
        |
        v
File-Type Detection
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
Knowledgebase Validation
        |
        v
Retrieval-Ready Dataset
(output/chunks.jsonl)
        |
        v
Sentence Transformer Model
        |
        v
384-Dimensional Embeddings
        |
        +---------------------+
        |                     |
        v                     v
     ChromaDB                FAISS
        |                     |
        +----------+----------+
                   |
                   v
          Semantic Retrieval
                   |
                   v
          Top-K Relevant Chunks
                   |
                   v
       Source Metadata + Scores
```

---

## 4. Day 1 — Clinical Knowledgebase Preparation

The first stage focuses on converting raw clinical documents into retrieval-ready datasets.

### 4.1 Multi-Format Document Processing

Supported formats:

- PDF
- DOCX
- TXT
- HTML / HTM
- PNG
- JPG / JPEG
- TIFF
- BMP
- WEBP

### 4.2 Text Extraction

The application uses:

- PyMuPDF for digital PDFs
- python-docx for Word documents
- BeautifulSoup for HTML documents
- Tesseract OCR for scanned documents and images

Tesseract must be installed separately for OCR functionality.

### 4.3 Logical Section Detection

Clinical documents are divided into meaningful sections where their structure can be identified.

Examples:

- Assessment
- Diagnosis
- Monitoring
- Treatment
- Follow-up
- Contraindications
- Emergency Management

The sectioning process uses document headings and structural heuristics.

### 4.4 Structure-Aware Chunking

Instead of splitting documents only by fixed character counts, the application attempts to preserve section and sentence boundaries.

Processing workflow:

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
Retrieval-Sized Chunks
```

Chunking is important because embedding models process limited amounts of text.

### 4.5 Metadata Preservation

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

Metadata helps maintain traceability between retrieved chunks and original clinical documents.

---

## 5. Day 1 — Knowledgebase Validation

The document processing pipeline was tested using two clinical guideline documents.

### Recorded Results

| Metric | Result |
|---|---|
| Documents discovered | 2 |
| Documents processed | 2 |
| Logical sections | 168 |
| Chunks generated | 631 |
| Failed documents | 0 |
| Average chunk size | 720.64 characters |
| Retrieval preparation status | READY |

These results confirm that the existing preprocessing pipeline generated a retrieval-ready dataset.

They do not independently establish the relevance or clinical correctness of future semantic search results.

---

## 6. Day 2 — Understanding Vector Embeddings

### 6.1 What Is a Vector Embedding?

A vector embedding is a numerical representation of data generated by an embedding model.

For text embeddings, the model converts text into numbers that encode learned semantic relationships.

Example:

```text
Text:
"Patients require blood pressure monitoring."

         |
         v

Embedding Model

         |
         v

[0.12, -0.34, 0.87, ..., 0.45]
```

The values shown above are illustrative.

Semantically related text often produces vectors that are close according to an appropriate similarity measure.

### 6.2 Embedding Model

Model used:

`sentence-transformers/all-MiniLM-L6-v2`

This model generates 384-dimensional embeddings.

For the existing 631 clinical text chunks, the expected embedding matrix is:

```text
Shape: (631, 384)
```

This means:

- 631 rows represent text chunks.
- 384 columns represent embedding dimensions.
- Total vector values: 242,304.

The actual output must be verified after successful embedding generation.

### 6.3 Embedding Generation

The embedding pipeline:

1. Reads `output/chunks.jsonl`.
2. Extracts valid chunk text.
3. Loads the embedding model.
4. Generates numerical embeddings.
5. Normalizes embeddings for similarity search.
6. Saves or passes the embeddings to the indexing process.

The same embedding model must be used for document vectors and query vectors.

### 6.4 Why Normalize Embeddings?

Vector normalization scales nonzero vectors to unit length.

For normalized embeddings, inner-product similarity is equivalent to cosine similarity.

This is useful when comparing search results across FAISS and ChromaDB.

---

## 7. Understanding Vector Databases

A vector database stores or manages numerical vector representations and supports similarity-based retrieval.

The typical workflow is:

```text
User Question
      |
      v
Query Embedding
      |
      v
Vector Similarity Search
      |
      v
Nearest Stored Vectors
      |
      v
Relevant Document Chunks
```

A vector search system may maintain:

- Unique vector IDs
- Vector embeddings
- Original text or references
- Document metadata
- Search indexes

### 7.1 Similarity Search

Important techniques include:

**Cosine similarity**

Measures directional similarity between vectors.

Higher cosine similarity generally indicates closer directional alignment.

**Euclidean distance**

Measures the straight-line distance between vectors.

Lower Euclidean distance indicates closer vectors.

**Inner product**

Measures the dot product between two vectors.

When both vectors are unit-normalized, inner product equals cosine similarity.

**Top-K retrieval**

Returns the K nearest search results.

For example, Top-K = 3 returns three matching chunks.

---

## 8. ChromaDB Implementation

ChromaDB is a vector database that supports collections, embeddings, document storage, metadata, and similarity queries.

### Features

- Vector storage
- Document storage
- Metadata management
- Similarity search
- Persistent local storage
- Collection-based organization

### Implementation Workflow

```text
chunks.jsonl
     |
     v
Embedding Model
     |
     v
Document Embeddings
     |
     v
ChromaDB Collection
     |
     v
Persistent Vector Storage
     |
     v
Semantic Search
```

The application uses generated embeddings to index clinical knowledge chunks.

Original source metadata is preserved where supported.

### ChromaDB Testing

The browser interface allows the user to:

1. Select an available chunk dataset.
2. Generate embeddings.
3. Build a ChromaDB index.
4. Enter a clinical search query.
5. Retrieve relevant document chunks.
6. Inspect scores and source information.

---

## 9. FAISS Implementation

FAISS (Facebook AI Similarity Search) is an open-source library for efficient vector similarity search.

It was developed by Meta AI.

Unlike ChromaDB, FAISS primarily provides vector indexing and nearest-neighbor search capabilities.

Original document text and metadata are managed separately by the application.

### FAISS Workflow

```text
Clinical Chunks
      |
      v
384-Dimensional Embeddings
      |
      v
FAISS Vector Index
      |
      v
Query Vector
      |
      v
Nearest-Neighbor Search
      |
      v
Matching Chunk IDs
      |
      v
Original Text + Metadata
```

### FAISS Testing

The application can:

1. Generate document vectors.
2. Create a FAISS index.
3. Add document embeddings.
4. Search for nearest vectors.
5. Map results back to original chunks.
6. Display source passages.

---

## 10. ChromaDB vs FAISS

| Feature | ChromaDB | FAISS |
|---|---|---|
| Technology | Vector database | Vector search library |
| Embedding storage | Supported | Supported through indexes |
| Original document storage | Built in | Requires separate management |
| Metadata | Built in | Managed separately |
| Similarity search | Supported | Supported |
| Local persistence | Supported | Index serialization |
| Collection management | Supported | Managed by application |
| Primary use | Document retrieval applications | Efficient vector similarity search |

### Learning Outcome

ChromaDB provides a higher-level database interface.

FAISS provides specialized vector indexing and similarity search capabilities.

Both can support semantic retrieval systems.

---

## 11. Browser-Based Streamlit Application

The Streamlit interface integrates the clinical document preparation workflow and vector search functionality.

### Day 1 Features

- Upload clinical documents
- Extract and organize content
- Generate logical sections
- Create retrieval chunks
- Validate the dataset
- Preview extracted chunks
- Review metadata
- Download generated files

### Day 2 Features

The updated application is designed to support:

- Loading existing clinical chunks
- Generating vector embeddings
- Building ChromaDB indexes
- Building FAISS indexes
- Performing semantic searches
- Selecting Top-K retrieval values
- Comparing retrieval results
- Viewing retrieved clinical text and source information

---

## 12. Project Structure

```text
clinical-knowledgebase-preparation/
|
|-- app.py
|-- build_knowledgebase.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- kb/
|   |-- __init__.py
|   |-- models.py
|   |-- extractors.py
|   |-- sectioning.py
|   |-- chunking.py
|   |-- validation.py
|   |-- builder.py
|
|-- vector_store/
|   |-- __init__.py
|   |-- embeddings.py
|
|-- input_docs/
|
|-- output/
|   |-- chunks.jsonl
|   |-- chunks.csv
|   |-- documents.json
|   |-- sections.json
|   |-- build_report.json
|   |-- embeddings.npy
|
|-- vector_data/
|   |-- chroma/
|   |-- faiss/
|
|-- tests/
```

Some generated directories and testing files are created only after the relevant operations are executed.

The exact contents may vary depending on the implementation.

---

## 13. Installation

### Step 1 — Clone Repository

```bash
git clone https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation.git
```

```bash
cd clinical-knowledgebase-preparation
```

### Step 2 — Create Virtual Environment

```powershell
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Additional packages required for vector search:

```powershell
python -m pip install sentence-transformers chromadb faiss-cpu numpy
```

### Main Technologies

- Python
- Streamlit
- Pandas
- PyMuPDF
- python-docx
- BeautifulSoup
- Tesseract OCR
- Pillow
- Sentence Transformers
- Hugging Face
- NumPy
- ChromaDB
- FAISS
- Git and GitHub

---

## 14. Hugging Face Model Cache Configuration

The embedding model is downloaded from Hugging Face when it is first used.

On Windows, the default cache location may be on the C: drive.

If storage space is limited, the cache can be moved to another drive.

Example for PowerShell:

```powershell
New-Item -ItemType Directory -Force "D:\huggingface_cache"

$env:HF_HOME = "D:\huggingface_cache"
$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"
```

These settings apply to the current terminal session.

The destination drive must have sufficient free storage.

---

## 15. Run the Application

Start the Streamlit interface:

```powershell
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### Build a Knowledgebase

1. Upload one or more clinical documents.
2. Click "Build Clinical Knowledgebase".
3. Review build statistics.
4. Inspect generated chunks.
5. Check metadata and validation results.
6. Download output files if required.

### Run Vector Database Indexing

1. Open the vector database section.
2. Select `Existing output/chunks.jsonl`.
3. Click the embedding and indexing button.
4. Generate embeddings.
5. Build ChromaDB and FAISS indexes.
6. Verify successful indexing.

### Test Semantic Search

1. Enter a question related to the uploaded documents.
2. Select ChromaDB, FAISS, or comparison mode.
3. Set Top-K to 3.
4. Execute semantic search.
5. Inspect retrieved passages.
6. Verify source metadata.
7. Compare retrieval rankings.

Example query:

```text
How should blood pressure be monitored?
```

The search query should be relevant to the actual contents of the indexed clinical documents.

---

## 16. Browser Testing Checklist

The following checks should be completed before considering Day 2 fully verified.

- [ ] Streamlit launches successfully.
- [ ] Existing clinical document processing works.
- [ ] Existing chunks.jsonl can be loaded.
- [ ] Embedding model loads successfully.
- [ ] Embeddings have 384 dimensions.
- [ ] ChromaDB index is created.
- [ ] FAISS index is created.
- [ ] Both indexes contain the expected number of vectors.
- [ ] Semantic search returns relevant passages.
- [ ] Retrieved chunk IDs map to the correct source text.
- [ ] Source metadata is displayed correctly.
- [ ] Top-K search returns the requested number of results when available.
- [ ] Saved indexes work after restarting the application.
- [ ] ChromaDB and FAISS results can be compared.

These checks must be verified by running the application; they should not be considered passed merely because the code exists.

---

## 17. GitHub Version Control

The project is maintained using Git and GitHub.

### Recommended Development Branch

```powershell
git switch -c feature/vector-databases
```

If the branch already exists, switch to it instead:

```powershell
git switch feature/vector-databases
```

### Check Changes

```powershell
git status
```

### Stage Project Updates

```powershell
git add app.py README.md requirements.txt .gitignore vector_store/
```

Review staged changes:

```powershell
git diff --cached --name-only
```

### Commit

```powershell
git commit -m "Add vector embeddings and semantic search with ChromaDB and FAISS"
```

### Push

```powershell
git push -u origin feature/vector-databases
```

After pushing, create a GitHub pull request to merge the feature branch into `main`.

---

## 18. Data Privacy and Git Security

Clinical source documents may contain sensitive information.

The following should not be committed without appropriate authorization:

- Patient information
- Clinical documents containing private data
- Generated chunks containing sensitive data
- Local ChromaDB databases
- FAISS index files
- Model caches
- Virtual environments
- API tokens and secrets

Recommended ignored paths:

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

If any sensitive files have previously been committed, adding them to `.gitignore` does not remove them from Git history.

---

## 19. Current Limitations

### Document Processing

- PDF heading detection uses structural heuristics.
- Complex PDF layouts may require more advanced extraction.
- Scanned documents depend on OCR quality.
- Some medical tables may lose formatting.
- Chunking is primarily character-based.

### Vector Retrieval

- The embedding model may truncate text exceeding its token limit.
- Semantic similarity does not guarantee clinical relevance.
- Metadata and source alignment must be preserved.
- Ranking quality depends on the dataset and embedding model.
- ChromaDB and FAISS may differ in scoring and indexing behavior.
- Clinical retrieval evaluation requires labeled queries and relevant-document judgments.

### RAG

This project currently focuses on document preparation and semantic retrieval.

LLM-based answer generation, medical reasoning, and automated clinical decision support are outside the completed scope.

---

## 20. Future Improvements

Potential future development:

1. Retrieval evaluation using Precision@K and Recall@K.
2. Clinical-domain embedding models.
3. Hybrid keyword and vector search.
4. Metadata-based filtering.
5. Cross-encoder reranking.
6. Token-aware chunking.
7. Duplicate detection.
8. Improved PDF table extraction.
9. Retrieval-Augmented Generation.
10. Source-grounded answer generation.

Any future clinical answering functionality should include appropriate evaluation, safety controls, and human oversight.

---

## 21. Learning Outcomes

This project provides practical experience with:

- Clinical document ingestion
- Multi-format data preprocessing
- Optical Character Recognition
- Knowledgebase engineering
- Structure-aware chunking
- Metadata preservation
- Retrieval-ready dataset preparation
- Text vector embeddings
- Embedding dimensions and normalization
- Cosine similarity
- Nearest-neighbor search
- ChromaDB collections
- FAISS indexes
- Vector persistence
- Semantic retrieval
- Top-K search
- Retrieval result comparison
- Streamlit application development
- Python project modularization
- GitHub version control

---

## 22. Development Status

### Day 1 — Clinical Knowledgebase Preparation

Status: Implemented and successfully tested through the CLI.

Recorded results:

- 2 clinical documents
- 168 sections
- 631 chunks
- 0 failed documents
- Retrieval-ready output generated

### Day 2 — Vector Databases

Status: Implementation prepared; complete browser and runtime verification pending.

Scope:

- Sentence Transformer embeddings
- ChromaDB vector storage
- FAISS similarity indexing
- Semantic search
- Retrieval comparison
- Browser integration

---

## 23. Project Repository

GitHub:

https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation

---

## 24. Conclusion

The Clinical Knowledgebase Preparation project demonstrates how unstructured clinical documents can be transformed into organized, metadata-rich knowledge suitable for AI retrieval applications.

Day 1 establishes a reliable document preparation pipeline.

Day 2 extends the application toward semantic retrieval by introducing vector embeddings, ChromaDB, FAISS, and similarity search.

Together, these components provide a practical foundation for building and evaluating future Retrieval-Augmented Generation systems.
