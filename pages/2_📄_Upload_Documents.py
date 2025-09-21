"""
Streamlit page for uploading and managing PDF documents.
Handles ingestion into OpenSearch with embeddings for RAG search.
"""

import logging
import os
import time

import streamlit as st
from PyPDF2 import PdfReader

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE
from src.embeddings import generate_embeddings, get_embedding_model
from src.ingestion import (
    bulk_index_documents,
    create_index,
    delete_documents_by_document_name,
)
from src.opensearch import get_opensearch_client
from src.utils import chunk_text, setup_logging


# ---------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="Gen AI - Upload Documents", page_icon="📂")

# Custom CSS injection
st.markdown(
    """
    <style>
    body { background-color: #f0f8ff; color: #002B5B; }
    .sidebar .sidebar-content {
        background-color: #006d77;
        color: white;
        padding: 20px;
        border-right: 2px solid #003d5c;
    }
    .sidebar h2, .sidebar h4 { color: white; }
    .block-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .footer-text {
        font-size: 1.1rem;
        font-weight: bold;
        color: black;
        text-align: center;
        margin-top: 10px;
    }
    .stButton button {
        background-color: #118ab2;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover { background-color: #07a6c2; color: white; }
    .stButton.delete-button button {
        background-color: #e63946;
        color: white;
        font-size: 14px;
    }
    .stButton.delete-button button:hover { background-color: #ff4c4c; }
    h1, h2, h3, h4 { color: #006d77; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar content
# ---------------------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center;'>Gen AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<h4 style='text-align: center;'>Your Document Assistant</h4>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
    <div class="footer-text">
        Using Gen AI
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Core upload/render function
# ---------------------------------------------------------------------
def render_upload_page() -> None:
    """
    Render the PDF upload/management interface:
    - Load embedding models (if not already cached)
    - Upload new documents and index them into OpenSearch
    - Display and delete existing indexed documents
    """
    st.title("Upload Documents")

    # Placeholder spinner for model loading
    model_status = st.empty()
    if "embedding_models_loaded" not in st.session_state:
        with model_status:
            with st.spinner("Loading embedding model for document processing..."):
                get_embedding_model()
                st.session_state["embedding_models_loaded"] = True
        model_status.empty()
        logger.info("Embedding models loaded and cached.")

    # Ensure upload directory exists
    upload_dir = "uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)

    # Connect to OpenSearch
    with st.spinner("Connecting to OpenSearch..."):
        client = get_opensearch_client()
    create_index(client)

    # Initialize documents state
    st.session_state["documents"] = []

    # Fetch existing document names from OpenSearch
    agg_query = {
        "size": 0,
        "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 10000}}},
    }
    response = client.search(index=OPENSEARCH_INDEX, body=agg_query)
    buckets = response["aggregations"]["unique_docs"]["buckets"]
    indexed_docs = [bucket["key"] for bucket in buckets]
    logger.info("Fetched indexed document names from OpenSearch.")

    # Populate session with existing docs
    for doc_name in indexed_docs:
        file_path = os.path.join(upload_dir, doc_name)
        if os.path.exists(file_path):
            reader = PdfReader(file_path)
            raw_text = "".join(page.extract_text() for page in reader.pages)
            st.session_state["documents"].append(
                {"filename": doc_name, "content": raw_text, "file_path": file_path}
            )
        else:
            st.session_state["documents"].append(
                {"filename": doc_name, "content": "", "file_path": None}
            )
            logger.warning(f"File '{doc_name}' missing locally.")

    # Notify if any file was deleted
    if "deleted_file" in st.session_state:
        st.success(f"Deleted file '{st.session_state['deleted_file']}' successfully.")
        del st.session_state["deleted_file"]

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            for file in uploaded_files:
                if file.name in indexed_docs:
                    st.warning(f"The file '{file.name}' is already indexed.")
                    continue

                file_path = save_uploaded_file(file)
                reader = PdfReader(file_path)
                raw_text = "".join(page.extract_text() for page in reader.pages)

                # Chunk + embed text
                chunks = chunk_text(raw_text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)
                vectors = generate_embeddings(chunks)

                to_index = [
                    {
                        "doc_id": f"{file.name}_{i}",
                        "text": chunk,
                        "embedding": vector,
                        "document_name": file.name,
                    }
                    for i, (chunk, vector) in enumerate(zip(chunks, vectors))
                ]
                bulk_index_documents(to_index)

                st.session_state["documents"].append(
                    {"filename": file.name, "content": raw_text, "file_path": file_path}
                )
                indexed_docs.append(file.name)
                logger.info(f"File '{file.name}' ingested successfully.")

        st.success("Upload and indexing complete!")

    # Display/manage documents
    if st.session_state["documents"]:
        st.markdown("### Uploaded Documents")
        with st.expander("Manage Uploaded Documents", expanded=True):
            for idx, doc in enumerate(st.session_state["documents"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(
                        f"{idx}. {doc['filename']} - {len(doc['content'])} characters extracted"
                    )
                with col2:
                    if st.button(
                        "Delete",
                        key=f"delete_{doc['filename']}_{idx}",
                        help=f"Delete {doc['filename']}",
                    ):
                        # Delete locally
                        if doc["file_path"] and os.path.exists(doc["file_path"]):
                            try:
                                os.remove(doc["file_path"])
                                logger.info(f"File '{doc['filename']}' removed from disk.")
                            except FileNotFoundError:
                                st.error(f"File '{doc['filename']}' not found on disk.")
                                logger.error(f"Failed to locate '{doc['filename']}' on delete.")
                        # Delete from index
                        delete_documents_by_document_name(doc["filename"])
                        st.session_state["documents"].pop(idx - 1)
                        st.session_state["deleted_file"] = doc["filename"]
                        time.sleep(0.5)
                        st.rerun()


# ---------------------------------------------------------------------
# File helper
# ---------------------------------------------------------------------
def save_uploaded_file(uploaded_file) -> str:  # type: ignore
    """
    Save an uploaded file to the `uploaded_files/` directory.
    """
    upload_dir = "uploaded_files"
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved file '{uploaded_file.name}' at '{file_path}'.")
    return file_path


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    render_upload_page()
