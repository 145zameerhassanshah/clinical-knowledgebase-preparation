"""Reusable in-memory clinical document → vector retrieval backend.

Source files and extracted text are not persisted by this workflow. It reuses
existing kb.builder.build_knowledgebase and returns only generated artifacts
for Streamlit session use. Not a clinical decision support tool.
"""
from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def build_from_uploads(files: list[tuple[str, bytes]]) -> dict:
    """Run existing Day 1 builder using temporary directories; return results."""
    from kb.builder import build_knowledgebase

    if not files:
        raise ValueError("Please upload at least one document.")
    with tempfile.TemporaryDirectory(prefix="clinical_workbench_") as tmp:
        inp = Path(tmp) / "input_docs"
        out = Path(tmp) / "output"
        inp.mkdir()
        out.mkdir()
        for i, (name, raw) in enumerate(files):
            safe_name = Path(name.replace("\\", "/")).name
            if not safe_name or safe_name in {".", ".."}:
                raise ValueError("Invalid filename")
            # Prefix prevents two same-name files from overwriting each other.
            (inp / f"{i:03d}_{safe_name}").write_bytes(raw)

        report = build_knowledgebase(str(inp), str(out))
        path = out / "chunks.jsonl"
        if not path.exists():
            raise RuntimeError("Builder did not produce chunks.jsonl")
        jsonl = path.read_bytes()
        rows = parse_chunks(jsonl)
        csv_file = out / "chunks.csv"
        return {
            "report": report,
            "rows": rows,
            "jsonl": jsonl,
            "csv": csv_file.read_bytes() if csv_file.exists() else b"",
            "source_count": len(files),
        }


def parse_chunks(raw: bytes) -> list[dict]:
    rows, used = [], set()
    for lineno, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {lineno}: {exc}") from exc
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        base_id = str(record.get("chunk_id") or f"chunk_{lineno}")
        chunk_id = base_id
        n = 2
        while chunk_id in used:
            chunk_id = f"{base_id}__{n}"
            n += 1
        used.add(chunk_id)
        rows.append({
            "id": chunk_id,
            "text": text.strip(),
            "source_file": str(record.get("source_file") or "unknown"),
            "section_title": str(record.get("section_title") or "unknown"),
            "document_id": str(record.get("document_id") or "unknown"),
            "source_type": str(record.get("source_type") or "unknown"),
        })
    if not rows:
        raise ValueError("No valid clinical text chunks found.")
    return rows


def encode(model, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts, batch_size=32, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    a = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if a.ndim != 2 or len(a) != len(texts) or not np.isfinite(a).all():
        raise ValueError("Invalid embeddings returned by model")
    return a


def make_indexes(model, rows: list[dict]):
    """Create FAISS exact + ephemeral Chroma indexes without saved patient files."""
    import chromadb
    import faiss

    vectors = encode(model, [r["text"] for r in rows])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Unique collection prevents colliding with a previous session.
    client = chromadb.EphemeralClient()
    collection_name = f"clinical_{uuid.uuid4().hex[:20]}"
    collection = client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
    for offset in range(0, len(rows), 100):
        batch = rows[offset:offset + 100]
        collection.add(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[{
                "source_file": r["source_file"],
                "section_title": r["section_title"],
                "document_id": r["document_id"],
                "source_type": r["source_type"],
            } for r in batch],
            embeddings=vectors[offset:offset + len(batch)].tolist(),
        )
    return {
        "faiss_index": index,
        "chroma_client": client,
        "chroma_collection": collection,
        "rows": rows,
        "vector_count": len(rows),
        "dimensions": vectors.shape[1],
        "model_id": MODEL_ID,
    }


def query_both(model, bundle: dict, question: str, k: int) -> dict:
    if not question.strip():
        raise ValueError("Please enter a search question")
    q = encode(model, [question.strip()])
    k = min(k, len(bundle["rows"]))
    scores, ids = bundle["faiss_index"].search(q, k)
    faiss_rows = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            row = bundle["rows"][int(idx)]
            faiss_rows.append({**row, "score": float(score)})

    collection = bundle["chroma_collection"]
    result = collection.query(
        query_embeddings=q.tolist(), n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chroma_rows = []
    for i, doc_id in enumerate(result["ids"][0]):
        md = result["metadatas"][0][i] or {}
        chroma_rows.append({
            "id": doc_id,
            "text": result["documents"][0][i],
            **md,
            "score": 1 - float(result["distances"][0][i]),
        })
    return {"FAISS Exact": faiss_rows, "ChromaDB": chroma_rows}
