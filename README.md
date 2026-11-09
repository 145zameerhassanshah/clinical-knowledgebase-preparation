# Clinical Knowledge Base Preparation

Continuation of the earlier multi-format clinical ingestion and chunking work.

## Pipeline
clinical documents → extraction → logical sections → structure-aware chunking → metadata propagation → retrieval-ready dataset

## Supported inputs
PDF, DOCX, TXT, HTML, images; scanned PDFs/images require local Tesseract OCR.

## Outputs
- output/documents.json
- output/sections.json
- output/chunks.jsonl
- output/chunks.csv
- output/build_report.json

## Run
```bash
pip install -r requirements.txt
python build_knowledgebase.py
```

Put your real clinical documents in `input_docs/`.

## Important
This milestone prepares data for retrieval. It does not yet generate embeddings or store vectors. Those are the next stages.
