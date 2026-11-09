# Clinical Knowledgebase Preparation

A multi-format clinical document processing and knowledge-base preparation system built with Python and Streamlit.

The project transforms heterogeneous clinical documents into structured, metadata-rich, retrieval-ready datasets that can later be used for semantic search, embeddings, vector databases, and Retrieval-Augmented Generation (RAG).

---

## Overview

Clinical information is commonly distributed across different document formats such as:

- PDF guidelines
- Word documents
- TXT files
- HTML pages
- scanned PDFs
- scanned images

These raw documents are not directly suitable for reliable semantic retrieval or RAG systems.

This project provides an end-to-end preparation pipeline that converts a collection of clinical documents into:

- logical document sections
- structure-aware chunks
- metadata-rich retrieval records
- source provenance
- JSONL datasets
- CSV inspection datasets
- processing and validation reports

The project can be used through both a command-line interface and a browser-based Streamlit application.

---

## Real-World Problem Solved

A healthcare organisation may have hundreds or thousands of clinical guidelines, protocols, SOPs, policies, and scanned documents.

Using these documents directly in an AI or RAG system can cause problems such as:

- inconsistent file formats
- missing document structure
- poor chunk boundaries
- lost source information
- weak retrieval quality
- difficulty tracing an answer back to its original document

This project converts raw documents into standardized knowledge units suitable for downstream AI systems.

```text
Clinical Documents
        ↓
File-Type Detection
        ↓
Text Extraction / OCR
        ↓
Logical Section Detection
        ↓
Structure-Aware Chunking
        ↓
Metadata & Source Preservation
        ↓
Validation
        ↓
Retrieval-Ready Dataset
```

---

## Features

### Multi-Format Document Processing

Supported formats include:

- PDF
- DOCX
- TXT
- HTML / HTM
- PNG
- JPG / JPEG
- TIFF
- BMP
- WEBP

---

### PDF Processing

Uses PyMuPDF for extracting text from digital PDFs.

For scan-like PDF pages, the pipeline can use Tesseract OCR as a fallback when Tesseract is installed locally.

---

### DOCX Processing

Uses `python-docx` to extract:

- paragraphs
- heading styles
- document metadata
- tables

Word heading styles are used to help preserve document hierarchy.

---

### HTML Processing

Uses BeautifulSoup to process:

- headings
- paragraphs
- list items
- page title

HTML structural elements such as `h1`, `h2`, and `h3` are used during section detection.

---

### OCR Support

Tesseract OCR can be used for:

- scanned PDFs
- PNG
- JPG
- TIFF
- other image-based clinical documents

Tesseract is an external local dependency and is not installed automatically by Python.

---

## Logical Section Detection

The pipeline attempts to preserve the original structure of each document before chunking.

Examples of logical clinical sections may include:

```text
Assessment
Diagnosis
Monitoring
Treatment
Follow-up
Contraindications
Emergency Management
```

Section detection uses format-specific signals.

### DOCX

Uses Word heading styles such as:

```text
Heading 1
Heading 2
Heading 3
```

### HTML

Uses:

```html
<h1>
<h2>
<h3>
```

### PDF / TXT

Uses conservative structural heuristics such as:

```text
1.1 Assessment
1.2 Diagnosis
MONITORING
FOLLOW-UP
```

---

## Structure-Aware Chunking

Instead of blindly splitting documents every fixed number of characters, the pipeline first identifies logical sections and then attempts to preserve sentence boundaries.

```text
Document
   ↓
Section
   ↓
Sentences
   ↓
Retrieval-Sized Chunks
```

This helps retain more meaningful context for future semantic retrieval.

The current implementation uses:

- logical section boundaries
- sentence boundaries
- target chunk size
- maximum chunk size
- limited sentence overlap

---

## Metadata Preservation

Every retrieval chunk retains information that allows the original source to be traced.

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

Metadata preservation is important for future:

- source citations
- document filtering
- auditability
- retrieval debugging
- RAG answer traceability

---

## Browser Interface

A Streamlit interface is included for easier interaction.

The browser workflow allows users to:

1. upload multiple clinical documents
2. build the knowledge base
3. monitor processing progress
4. review build statistics
5. inspect generated chunks
6. inspect document metadata
7. inspect detected sections
8. review validation results
9. download generated datasets

Run:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Command-Line Usage

The same backend can also be executed directly from the terminal.

Place clinical documents inside:

```text
input_docs/
```

Then run:

```bash
python build_knowledgebase.py
```

Example output:

```text
Clinical Knowledge Base Build Complete
---------------------------------------------
Documents discovered : 2
Documents processed  : 2
Sections created     : 168
Chunks created       : 631
Failed documents     : 0
Avg chunk chars      : 720.64
Retrieval ready      : YES
```

---

## Real Clinical Validation

The pipeline was tested using two real clinical guideline documents.

Results:

```text
Documents processed : 2
Logical sections    : 168
Chunks generated    : 631
Average chunk size  : 720.64 characters
Failed documents    : 0
Retrieval ready     : YES
```

These results demonstrate that the pipeline can process larger real-world clinical documents rather than only small demonstration files.

---

## Output Files

The system creates the following files inside the `output/` directory.

### `documents.json`

Contains document-level information such as:

- document ID
- filename
- source type
- extraction method
- metadata
- warnings
- extracted text statistics

---

### `sections.json`

Contains detected logical sections including:

- section ID
- document ID
- section title
- section level
- section content
- character count

---

### `chunks.jsonl`

Primary retrieval-ready dataset.

Each JSON line represents one chunk.

This format can later be streamed into an embedding pipeline or vector database.

---

### `chunks.csv`

Human-readable representation of generated chunks.

Useful for:

- manual inspection
- chunk quality review
- debugging
- spreadsheet analysis

---

### `build_report.json`

Contains build and validation statistics such as:

- number of discovered documents
- successfully processed documents
- sections created
- chunks generated
- failed documents
- average chunk size
- validation issues
- retrieval readiness

---

## Project Structure

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
├── input_docs/
│
└── output/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/145zameerhassanshah/clinical-knowledgebase-preparation.git
```

Then:

```bash
cd clinical-knowledgebase-preparation
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
streamlit
pandas
pymupdf
python-docx
beautifulsoup4
pytesseract
Pillow
```

---

## Tesseract OCR

OCR functionality requires the Tesseract OCR engine to be installed separately.

Typical Windows installation path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Check installation:

```powershell
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

Tesseract is required mainly for:

- image files
- scanned documents
- scan-like PDF pages

Digital PDFs can still be processed without OCR.

---

## Run the Browser Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Upload clinical documents and select:

```text
Build Clinical Knowledgebase
```

---

## Run Through CLI

Place files inside:

```text
input_docs/
```

Then run:

```bash
python build_knowledgebase.py
```

Generated datasets will appear inside:

```text
output/
```

---

## Retrieval-Ready vs RAG-Ready

This project prepares the knowledge layer for retrieval.

Current pipeline:

```text
Clinical Documents
        ↓
Extraction
        ↓
Sections
        ↓
Chunks
        ↓
Metadata
        ↓
Retrieval-Ready Dataset
```

The current project does not yet perform:

- embedding generation
- vector database indexing
- semantic search
- reranking
- LLM answer generation

Those belong to the next RAG stage.

---

## Planned Next Stage

The planned extension is:

```text
chunks.jsonl
      ↓
Embedding Model
      ↓
Vector Embeddings
      ↓
Vector Database
      ↓
Semantic Retriever
      ↓
Retrieval Evaluation
      ↓
RAG
      ↓
Grounded Answer + Source Citation
```

Potential future technologies include:

- Sentence Transformers
- Hugging Face embeddings
- FAISS
- ChromaDB
- Qdrant
- reranking models
- retrieval evaluation metrics

---

## Current Limitations

The current project intentionally focuses on knowledge-base preparation.

Known limitations include:

- PDF heading detection currently uses heuristic rules
- complex multi-column PDFs may require layout-aware parsing
- table reconstruction is limited
- OCR accuracy depends on scan quality
- handwritten text is not specifically optimized
- chunk size is currently character-based rather than tokenizer-based
- retrieval quality has not yet been evaluated using query/relevance datasets
- no embedding model or vector database is currently included

These limitations represent future engineering and research directions rather than hidden functionality.

---

## Key Learning Outcomes

This project explores:

- document ingestion architecture
- multi-format extraction
- OCR integration
- corpus preparation
- logical section detection
- structure-aware chunking
- metadata engineering
- source provenance
- batch processing
- knowledge-base preparation
- JSONL dataset design
- retrieval preparation
- RAG data engineering
- Streamlit frontend integration
- validation and observability

---

## Technologies

```text
Python
Streamlit
Pandas
PyMuPDF
python-docx
BeautifulSoup
Tesseract OCR
Pillow
JSON
JSONL
CSV
Git
GitHub
```

---

## Development Context

This project was developed as part of practical AI/ML and RAG engineering learning focused on converting heterogeneous real-world documents into reliable machine-retrievable knowledge.

The implementation emphasizes:

- modularity
- provenance
- transparent processing
- local execution
- retrieval readiness
- future extensibility

---

