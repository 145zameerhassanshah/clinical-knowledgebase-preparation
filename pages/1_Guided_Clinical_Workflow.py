"""Guided workflow page: upload → process → index → retrieve, no CLI required."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.clinical_workflow import (
    MODEL_ID, build_from_uploads, make_indexes, query_both,
)

st.set_page_config(page_title="Guided Clinical AI Workflow", page_icon="🧭", layout="wide")
st.title("🧭 Guided Clinical AI Workflow")
st.caption("One browser: upload → build knowledgebase → index vectors → search sources")
st.warning("Research / education only. Use public or appropriately de-identified guidelines, not private patient records. Retrieved passages are not medical advice.")

@st.cache_resource(show_spinner="Loading embedding model...")
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device="cpu")

state = st.session_state
state.setdefault("guided_build", None)
state.setdefault("guided_index", None)
state.setdefault("guided_search", None)

with st.expander("What is automatic here?", expanded=False):
    st.markdown("""**Upload & Process** builds chunks using your existing `kb.builder`.
    **Build Search Indexes** embeds those chunks using MiniLM and creates FAISS Exact + temporary ChromaDB indexes.
    **Search** compares matching passages and sources. Day 3's 32-pair evaluation and Day 4's 50,000-vector synthetic benchmark remain separate controlled experiments.
    This guided page does **not** rewrite `app.py` or alter existing persistent indexes.""")

st.subheader("Step 1 · Upload and prepare clinical guidelines")
uploaded = st.file_uploader(
    "Choose public guideline PDF, DOCX, TXT, HTML or image files",
    type=["pdf", "docx", "txt", "html", "htm", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    accept_multiple_files=True, key="guided_documents",
)
if uploaded:
    st.caption(f"{len(uploaded)} file(s) selected: " + ", ".join(f.name for f in uploaded[:6]))
if st.button("1 · Process uploaded documents", type="primary", disabled=not uploaded):
    try:
        with st.spinner("Extracting, sectioning, chunking and validating..."):
            data = build_from_uploads([(f.name, f.getvalue()) for f in uploaded])
        # Old indexes must not be reused for newly uploaded documents.
        state.guided_build = data
        state.guided_index = None
        state.guided_search = None
        st.success("Knowledgebase built successfully.")
    except ImportError as exc:
        st.error(f"Required package missing: {exc}. Install dependencies from requirements.txt.")
    except Exception as exc:
        st.error(f"Document processing failed: {exc}")

build = state.guided_build
if build is not None:
    report = build["report"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Uploaded documents", build["source_count"])
    c2.metric("Valid chunks", len(build["rows"]))
    c3.metric("Retrieval ready", "YES" if report.get("retrieval_ready") else "REVIEW")
    with st.expander("Document processing report"):
        st.json(report)
    st.dataframe(pd.DataFrame(build["rows"][:20]), hide_index=True, use_container_width=True)
    st.download_button("Download chunks.jsonl", build["jsonl"], "chunks.jsonl", mime="application/json")
    if build["csv"]:
        st.download_button("Download chunks.csv", build["csv"], "chunks.csv", mime="text/csv")

st.divider()
st.subheader("Step 2 · Generate embeddings and create vector indexes")
st.caption(f"Model: `{MODEL_ID}` · 384-dimensional normalized vectors")
if st.button("2 · Build FAISS + ChromaDB indexes", disabled=build is None, type="primary"):
    try:
        with st.spinner("Loading model, generating embeddings and building indexes..."):
            bundle = make_indexes(embedding_model(), build["rows"])
        state.guided_index = bundle
        state.guided_search = None
        st.success(f"Indexed {bundle['vector_count']} chunks, {bundle['dimensions']} dimensions.")
    except ImportError as exc:
        st.error(f"Missing dependency: {exc}. Try: python -m pip install sentence-transformers chromadb faiss-cpu")
    except Exception as exc:
        st.error(f"Indexing failed: {exc}")

bundle = state.guided_index
if bundle is not None:
    a, b, c = st.columns(3)
    a.metric("Indexed vectors", bundle["vector_count"])
    b.metric("Dimensions", bundle["dimensions"])
    c.metric("Search backends", "2")

st.divider()
st.subheader("Step 3 · Search and compare real document passages")
with st.form("guided_search_form"):
    question = st.text_input("Ask about your uploaded guidelines", placeholder="How should blood pressure be monitored?")
    k = st.slider("Top-K passages", 1, 10, 3)
    submitted = st.form_submit_button("3 · Search both vector stores", disabled=bundle is None)
if submitted and bundle is not None:
    try:
        with st.spinner("Searching... "):
            state.guided_search = query_both(embedding_model(), bundle, question, k)
    except Exception as exc:
        st.error(f"Search failed: {exc}")

if state.guided_search:
    left, right = st.columns(2)
    for col, backend in zip([left, right], ["ChromaDB", "FAISS Exact"]):
        with col:
            st.markdown(f"#### {backend}")
            for rank, item in enumerate(state.guided_search[backend], 1):
                with st.expander(f"#{rank} · {item.get('section_title', 'Unknown')} · score {item['score']:.3f}", expanded=(rank == 1)):
                    st.write(item["text"])
                    st.caption(f"Source: {item.get('source_file', 'unknown')} | ID: {item['id']}")
    ids_1 = {x["id"] for x in state.guided_search["ChromaDB"]}
    ids_2 = {x["id"] for x in state.guided_search["FAISS Exact"]}
    st.caption(f"Shared returned chunk IDs: {len(ids_1 & ids_2)} / {min(len(ids_1), len(ids_2))}. Chroma uses approximate HNSW search; ranks may differ.")

st.divider()
st.subheader("Step 4 · Explore the controlled evaluation experiments")
st.write("These experiments don't automatically use the just-uploaded documents. They evaluate model behavior and vector-search performance on their own benchmark datasets.")
page3 = ROOT / "pages" / "3_Embedding_Model_Evaluation.py"
page4 = ROOT / "pages" / "4_Vector_Search_Performance.py"
a, b = st.columns(2)
with a:
    st.markdown("**Day 3 — Model comparison**")
    st.write("32 medical sentence pairs, cosine scores, PCA")
    if page3.exists():
        st.page_link("pages/3_Embedding_Model_Evaluation.py", label="Open embedding evaluation", icon="🧬")
    else:
        st.info("Add pages/3_Embedding_Model_Evaluation.py to enable Day 3.")
with b:
    st.markdown("**Day 4 — Performance benchmark**")
    st.write("50,000 synthetic vectors, NumPy vs FAISS, Recall@10")
    if page4.exists():
        st.page_link("pages/4_Vector_Search_Performance.py", label="Open vector performance lab", icon="⚡")
    else:
        st.info("Add pages/4_Vector_Search_Performance.py to enable Day 4.")

st.caption("Indexes created by this guided workflow are held in the current application session. Rebuild after restarting Streamlit. Day 2's existing persistent-index workflow remains available on the main page.")
