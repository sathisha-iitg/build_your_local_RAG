"""
Embedding utilities for converting text into vector representations.

This module wraps SentenceTransformers to load the model once,
and generate embeddings for downstream semantic search.
"""

import logging
from typing import Any, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import setup_logging

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model and cache it across sessions.

    Returns:
        SentenceTransformer: Pretrained embedding model instance.
    """
    log.info("Loading embedding model from: %s", EMBEDDING_MODEL_PATH)
    return SentenceTransformer(EMBEDDING_MODEL_PATH)


# ---------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------
def generate_embeddings(chunks: List[str]) -> List[np.ndarray[Any, Any]]:
    """
    Generate vector embeddings for a list of text segments.

    Args:
        chunks (List[str]): Text segments to encode.

    Returns:
        List[np.ndarray[Any, Any]]: Embedding vectors for each segment.
    """
    model = get_embedding_model()
    vectors = [np.array(model.encode(segment)) for segment in chunks]
    log.info("Generated embeddings for %d chunks.", len(chunks))
    return vectors
