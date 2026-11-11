from pathlib import Path
import io
import json
import tempfile
import zipfile

import pandas as pd
import streamlit as st

from kb.builder import build_knowledgebase


st.set_page_config(
    page_title="Clinical Knowledgebase Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --bg:#07111f;
    --panel:#0d1b2d;
    --line:rgba(148,163,184,.16);
    --text:#eef5ff;
    --muted:#9fb0c7;
    --accent:#4ea8ff;
    --success:#43d19e;
}

[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 10% 5%, rgba(78,168,255,.12), transparent 24%),
      radial-gradient(circle at 85% 0%, rgba(124,156,255,.08), transparent 20%),
      var(--bg);
}

[data-testid="stSidebar"] {
    background:#081524;
    border-right:1px solid var(--line);
}

.block-container {
    max-width:1450px;
    padding-top:1.2rem;
    padding-bottom:4rem;
}

.hero {
    border:1px solid var(--line);
    border-radius:24px;
    padding:28px 30px;
    background:linear-gradient(
        135deg,
        rgba(16,35,59,.98),
        rgba(9,25,43,.96)
    );
    box-shadow:0 24px 70px rgba(0,0,0,.20);
    margin-bottom:22px;
}

.hero-kicker {
    color:#78bcff;
    font-weight:800;
    font-size:.76rem;
    letter-spacing:.14em;
    text-transform:uppercase;
}

.hero h1 {
    margin:8px 0 8px;
    font-size:2.45rem;
    letter-spacing:-.04em;
}

.hero p {
    color:var(--muted);
    max-width:850px;
    line-height:1.6;
    margin-bottom:0;
}

.badge {
    display:inline-block;
    margin-top:14px;
    padding:6px 11px;
    border-radius:999px;
    border:1px solid rgba(67,209,158,.25);
    background:rgba(67,209,158,.08);
    color:#76e7be;
    font-size:.78rem;
    font-weight:800;
}

.section-title {
    margin-top:20px;
    margin-bottom:8px;
    font-size:1.2rem;
    font-weight:800;
}

div[data-testid="stFileUploader"] {
    border:1px dashed rgba(78,168,255,.35);
    border-radius:18px;
    padding:8px;
    background:rgba(13,27,45,.52);
}

.stButton > button,
.stDownloadButton > button {
    border-radius:11px;
    border:1px solid rgba(78,168,255,.25);
    font-weight:700;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


st.sidebar.markdown("## 🧠 Clinical KB Builder")
st.sidebar.caption("Retrieval-ready clinical knowledge preparation")

st.sidebar.divider()

st.sidebar.markdown("**Pipeline**")
st.sidebar.caption(
    "Upload → Extract → Section → Chunk → Validate → Export"
)

st.sidebar.markdown("**Supported**")
st.sidebar.caption(
    "PDF · DOCX · TXT · HTML · Images / scans"
)

st.sidebar.markdown("**Status**")
st.sidebar.caption("🟢 Knowledgebase milestone")


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">
        Clinical Knowledge Engineering
      </div>

      <h1>
        Clinical Knowledgebase Builder
      </h1>

      <p>
        Transform a collection of clinical documents into
        structured sections, metadata-rich chunks and a
        unified retrieval-ready dataset for downstream
        semantic search and RAG.
      </p>

      <span class="badge">
        ● Local-first processing
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)


SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "txt",
    "html",
    "htm",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
    "webp",
]


uploaded = st.file_uploader(
    "Upload clinical documents",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help=(
        "Upload one or more clinical documents "
        "to build a unified knowledge base."
    ),
)


st.caption(
    "Scanned images/PDFs require "
    "Tesseract OCR installed locally."
)


if uploaded:

    st.info(
        f"{len(uploaded)} document(s) selected."
    )

    if st.button(
        "Build Clinical Knowledgebase",
        type="primary",
        use_container_width=True,
    ):

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            input_dir = tmp / "input_docs"
            output_dir = tmp / "output"

            input_dir.mkdir()
            output_dir.mkdir()

            for file in uploaded:

                target = input_dir / file.name

                target.write_bytes(
                    file.getbuffer()
                )

            progress = st.progress(
                0,
                text="Preparing documents..."
            )

            try:

                progress.progress(
                    20,
                    text=(
                        "Extracting and organizing "
                        "document content..."
                    ),
                )

                report = build_knowledgebase(
                    str(input_dir),
                    str(output_dir),
                )

                progress.progress(
                    80,
                    text="Validating retrieval dataset...",
                )

                documents_path = (
                    output_dir / "documents.json"
                )

                sections_path = (
                    output_dir / "sections.json"
                )

                chunks_jsonl = (
                    output_dir / "chunks.jsonl"
                )

                # Retain JSONL in session memory after the temporary directory closes.
                st.session_state["current_kb_jsonl"] = chunks_jsonl.read_bytes()
                st.session_state["kb_source_label"] = "Just-built clinical documents"

                chunks_csv = (
                    output_dir / "chunks.csv"
                )

                report_path = (
                    output_dir / "build_report.json"
                )

                documents = json.loads(
                    documents_path.read_text(
                        encoding="utf-8"
                    )
                )

                sections = json.loads(
                    sections_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    chunks_csv.exists()
                    and chunks_csv.stat().st_size > 0
                ):

                    df = pd.read_csv(
                        chunks_csv
                    )

                else:

                    df = pd.DataFrame()

                progress.progress(
                    100,
                    text="Knowledgebase ready.",
                )

                if report["retrieval_ready"]:

                    st.success(
                        "Knowledgebase build "
                        "completed successfully."
                    )

                else:

                    st.warning(
                        "Build completed, but review "
                        "the report before retrieval use."
                    )


                st.markdown(
                    '<div class="section-title">'
                    'Build Summary'
                    '</div>',
                    unsafe_allow_html=True,
                )


                c1, c2, c3, c4, c5 = st.columns(5)

                c1.metric(
                    "Documents",
                    report["documents_processed"],
                )

                c2.metric(
                    "Sections",
                    report["sections_created"],
                )

                c3.metric(
                    "Chunks",
                    report["chunks_created"],
                )

                c4.metric(
                    "Avg chars",
                    report["average_chunk_chars"],
                )

                c5.metric(
                    "Status",
                    (
                        "READY"
                        if report["retrieval_ready"]
                        else "CHECK"
                    ),
                )


                st.markdown(
                    '<div class="section-title">'
                    'Quality Check'
                    '</div>',
                    unsafe_allow_html=True,
                )


                q1, q2, q3 = st.columns(3)

                q1.metric(
                    "Failed docs",
                    report["failed_documents"],
                )

                q2.metric(
                    "Smallest chunk",
                    report.get(
                        "smallest_chunk_chars",
                        0,
                    ),
                )

                q3.metric(
                    "Largest chunk",
                    report.get(
                        "largest_chunk_chars",
                        0,
                    ),
                )


                if report.get(
                    "validation_issue_count",
                    0,
                ):

                    st.warning(
                        f"{report['validation_issue_count']} "
                        "validation issue(s) found."
                    )

                    st.json(
                        report.get(
                            "validation_issues",
                            [],
                        )
                    )

                else:

                    st.success(
                        "No validation issues detected."
                    )


                st.markdown(
                    '<div class="section-title">'
                    'Chunk Preview'
                    '</div>',
                    unsafe_allow_html=True,
                )


                if not df.empty:

                    preview_columns = [
                        column
                        for column in [
                            "chunk_id",
                            "section_title",
                            "text",
                            "source_file",
                            "source_type",
                            "char_count",
                        ]
                        if column in df.columns
                    ]

                    st.dataframe(
                        df[
                            preview_columns
                        ].head(50),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.info(
                        "No chunks available "
                        "for preview."
                    )


                with st.expander(
                    "Document metadata"
                ):

                    st.json(
                        documents
                    )


                with st.expander(
                    "Detected sections"
                ):

                    st.json(
                        sections[:30]
                    )

                    if len(sections) > 30:

                        st.caption(
                            f"Showing 30 of "
                            f"{len(sections)} sections."
                        )


                with st.expander(
                    "Build report"
                ):

                    st.json(
                        report
                    )


                st.markdown(
                    '<div class="section-title">'
                    'Download Outputs'
                    '</div>',
                    unsafe_allow_html=True,
                )


                d1, d2, d3 = st.columns(3)


                with d1:

                    st.download_button(
                        "Download chunks.jsonl",
                        chunks_jsonl.read_bytes(),
                        "chunks.jsonl",
                        "application/json",
                        use_container_width=True,
                    )


                with d2:

                    st.download_button(
                        "Download chunks.csv",
                        chunks_csv.read_bytes(),
                        "chunks.csv",
                        "text/csv",
                        use_container_width=True,
                    )


                with d3:

                    st.download_button(
                        "Download build report",
                        report_path.read_bytes(),
                        "build_report.json",
                        "application/json",
                        use_container_width=True,
                    )


                zip_buffer = io.BytesIO()


                with zipfile.ZipFile(
                    zip_buffer,
                    "w",
                    zipfile.ZIP_DEFLATED,
                ) as z:

                    for path in output_dir.iterdir():

                        if path.is_file():

                            z.write(
                                path,
                                arcname=path.name,
                            )


                zip_buffer.seek(0)


                st.download_button(
                    "Download complete knowledgebase package",
                    zip_buffer.getvalue(),
                    "clinical_knowledgebase_output.zip",
                    "application/zip",
                    use_container_width=True,
                )


            except Exception as exc:

                st.error(
                    f"Build failed: {exc}"
                )


else:

    st.markdown(
        '<div class="section-title">'
        'What this tool produces'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - **documents.json** — document-level metadata
          and source provenance
        - **sections.json** — logical document sections
        - **chunks.jsonl** — primary retrieval-ready dataset
        - **chunks.csv** — human-readable chunk inspection
        - **build_report.json** — processing and validation statistics
        """
    )


st.divider()

st.caption(
    "Clinical Knowledgebase Builder • "
    "Streamlit UI • "
    "Retrieval-ready data preparation"
)

# ================================================================
# DAY 2: Vector Embeddings, ChromaDB, FAISS and Semantic Search
# Preserves all of the original document preparation functionality.
# ================================================================

import os
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_ROOT = PROJECT_ROOT / "vector_data"
CHROMA_DIR = VECTOR_ROOT / "chroma"
FAISS_INDEX_FILE = VECTOR_ROOT / "clinical.index"
FAISS_DOCS_FILE = VECTOR_ROOT / "faiss_documents.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "output" / "chunks.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_jsonl(raw: bytes):
    """Return clean unique chunk records and preserve source provenance."""
    rows = []
    seen = set()
    for line_no, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {line_no}: {exc}") from exc
        if not isinstance(chunk, dict):
            raise ValueError(f"JSONL line {line_no} must contain an object")
        text_value = chunk.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            continue
        original_id = str(chunk.get("chunk_id") or f"chunk_{line_no:06d}")
        chunk_id = original_id
        suffix = 2
        while chunk_id in seen:
            chunk_id = f"{original_id}__{suffix}"
            suffix += 1
        seen.add(chunk_id)
        rows.append({
            "id": chunk_id,
            "text": text_value.strip(),
            "source_file": str(chunk.get("source_file") or "unknown"),
            "section_title": str(chunk.get("section_title") or "unknown"),
            "document_id": str(chunk.get("document_id") or "unknown"),
            "source_type": str(chunk.get("source_type") or "unknown"),
        })
    if not rows:
        raise ValueError("No non-empty text chunks found in JSONL")
    return rows


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def encode_texts(texts):
    model = load_embedding_model()
    return np.asarray(
        model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )


def build_vector_indexes(rows):
    import chromadb
    import faiss

    VECTOR_ROOT.mkdir(parents=True, exist_ok=True)
    vectors = encode_texts([row["text"] for row in rows])
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError("Unexpected embedding shape")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain NaN or infinity")

    # ChromaDB keeps the text, IDs and metadata next to its index.
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = "clinical_chunks"
    try:
        chroma_client.delete_collection(name=collection_name)
    except chromadb.errors.NotFoundError:
        pass
    collection = chroma_client.create_collection(
        name=collection_name,
        configuration={"hnsw": {"space": "cosine"}},
    )
    for start in range(0, len(rows), 100):
        batch = rows[start : start + 100]
        collection.add(
            ids=[r["id"] for r in batch],
            documents=[r["text"] for r in batch],
            metadatas=[{
                "source_file": r["source_file"],
                "section_title": r["section_title"],
                "document_id": r["document_id"],
                "source_type": r["source_type"],
            } for r in batch],
            embeddings=vectors[start : start + len(batch)].tolist(),
        )

    # FAISS indexes the vectors; document metadata remains in a separate file.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    FAISS_DOCS_FILE.write_text(
        json.dumps({"model": EMBED_MODEL_NAME, "rows": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(rows), vectors.shape[1]


def search_chroma(query_vector, k):
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name="clinical_chunks")
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for i, doc_id in enumerate(result["ids"][0]):
        matches.append({
            "id": doc_id,
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i] or {},
            "score": 1.0 - float(result["distances"][0][i]),
        })
    return matches


def search_faiss(query_vector, k):
    import faiss
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    saved = json.loads(FAISS_DOCS_FILE.read_text(encoding="utf-8"))
    if saved.get("model") != EMBED_MODEL_NAME:
        raise ValueError("FAISS index was created using a different embedding model")
    rows = saved["rows"]
    if len(rows) != index.ntotal:
        raise ValueError("FAISS metadata and index contain different record counts")
    scores, indices = index.search(
        np.ascontiguousarray(query_vector), min(k, index.ntotal)
    )
    matches = []
    for score, row_index in zip(scores[0], indices[0]):
        if row_index < 0:
            continue
        row = rows[int(row_index)]
        matches.append({
            "id": row["id"],
            "text": row["text"],
            "metadata": row,
            "score": float(score),
        })
    return matches


def show_matches(matches):
    if not matches:
        st.info("No matches found.")
        return
    for rank, match in enumerate(matches, start=1):
        md = match["metadata"]
        with st.expander(
            f"#{rank} · {md.get('section_title', 'Unknown section')} "
            f"· similarity {match['score']:.3f}",
            expanded=(rank == 1),
        ):
            st.write(match["text"])
            st.caption(
                f"Source: {md.get('source_file', 'unknown')} | "
                f"Chunk: {match['id']} | "
                f"Document: {md.get('document_id', 'unknown')}"
            )


st.divider()
st.header("🔎 Day 2 · Vector Databases & Semantic Search")
st.caption(
    "Generate embeddings from clinical chunks, create persistent ChromaDB / FAISS "
    "indexes, and compare semantic retrieval in the browser. "
    "Search results are source passages, not medical advice."
)

st.warning(
    "Use public or appropriately de-identified documents. Indexes save text and "
    "metadata locally under vector_data/. Do not commit patient data or indexes to Git."
)

source_choice = st.radio(
    "Choose a chunk dataset",
    ["Current browser build", "Existing output/chunks.jsonl", "Upload chunks.jsonl"],
    horizontal=True,
)

raw_jsonl = None
if source_choice == "Current browser build":
    raw_jsonl = st.session_state.get("current_kb_jsonl")
    if raw_jsonl is None:
        st.info("Build a knowledgebase above first, or select existing output/chunks.jsonl.")
elif source_choice == "Existing output/chunks.jsonl":
    if DEFAULT_CHUNKS.exists():
        raw_jsonl = DEFAULT_CHUNKS.read_bytes()
    else:
        st.info("No output/chunks.jsonl found. Run python build_knowledgebase.py first.")
else:
    uploaded_jsonl = st.file_uploader(
        "Select a prepared chunks.jsonl file",
        type=["jsonl"],
        key="vector_jsonl_upload",
    )
    if uploaded_jsonl is not None:
        raw_jsonl = uploaded_jsonl.getvalue()

if raw_jsonl is not None:
    try:
        parsed_rows = parse_jsonl(raw_jsonl)
        c1, c2 = st.columns(2)
        c1.metric("Valid text chunks", len(parsed_rows))
        c2.metric("Embedding dimensions (model)", 384)
        with st.expander("Preview chunks used for indexing"):
            st.dataframe(pd.DataFrame(parsed_rows[:10]), hide_index=True)

        if st.button("Generate embeddings + build both vector indexes", type="primary"):
            with st.spinner("Loading embedding model and indexing clinical chunks..."):
                count, dimensions = build_vector_indexes(parsed_rows)
            st.session_state["last_index_details"] = (count, dimensions)
            st.success(
                f"Indexed {count} chunks with {dimensions}-dimensional embeddings "
                "in ChromaDB and FAISS."
            )
    except Exception as exc:
        st.error(f"Cannot process chunks / build indexes: {exc}")
        if "space" in str(exc).lower() or "disk" in str(exc).lower():
            st.info(
                "Disk space issue: set HF_HOME and HF_HUB_CACHE to a roomy D: drive "
                "folder in PowerShell before launching Streamlit."
            )

has_chroma = CHROMA_DIR.exists()
has_faiss = FAISS_INDEX_FILE.exists() and FAISS_DOCS_FILE.exists()
if has_chroma or has_faiss:
    st.subheader("Test semantic search")
    with st.form("vector_search_form"):
        question = st.text_input(
            "Clinical document search query",
            placeholder="e.g. What are the guidelines for monitoring blood pressure?",
        )
        backend = st.selectbox(
            "Search backend",
            ["Compare both", "ChromaDB", "FAISS"],
        )
        k = st.slider("Top-K matching chunks", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Embedding your query and finding nearest chunks..."):
                    query_vector = encode_texts([question.strip()])
                    chroma_results = (
                        search_chroma(query_vector, k)
                        if backend in ("Compare both", "ChromaDB") else None
                    )
                    faiss_results = (
                        search_faiss(query_vector, k)
                        if backend in ("Compare both", "FAISS") else None
                    )
                if backend == "Compare both":
                    left, right = st.columns(2)
                    with left:
                        st.markdown("#### ChromaDB")
                        show_matches(chroma_results)
                    with right:
                        st.markdown("#### FAISS")
                        show_matches(faiss_results)
                elif backend == "ChromaDB":
                    show_matches(chroma_results)
                else:
                    show_matches(faiss_results)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
else:
    st.info("Build both vector indexes above to enable semantic search.")
