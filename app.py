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