import logging
from typing import Dict, Iterable, List, Optional

import ollama
import streamlit as st

from src.constants import ASSYMETRIC_EMBEDDING, OLLAMA_MODEL_NAME
from src.embeddings import get_embedding_model
from src.opensearch import hybrid_search
from src.utils import setup_logging

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_model_pulled(model: str) -> bool:
    """
    Make sure the given model is available locally; pull it if missing.

    Args:
        model (str): Name of the model to verify.

    Returns:
        bool: True if model is ready, False on failure.
    """
    try:
        existing = ollama.list()
        if model not in existing:
            log.info("Model '%s' not found locally. Pulling from registry...", model)
            ollama.pull(model)
            log.info("Model '%s' successfully pulled and ready.", model)
        else:
            log.info("Model '%s' already available locally.", model)
    except ollama.ResponseError as exc:
        log.error("Failed to check/pull model: %s", exc.error)
        return False
    return True


# ---------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------
def run_llama_streaming(prompt: str, temperature: float) -> Optional[Iterable[str]]:
    """
    Stream responses from the Ollama LLaMA model.

    Args:
        prompt (str): Input prompt for the model.
        temperature (float): Sampling temperature.

    Returns:
        Optional[Iterable[str]]: Stream of response chunks, or None on error.
    """
    try:
        log.info("Initiating response stream from LLaMA model...")
        return ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature},
        )
    except ollama.ResponseError as exc:
        log.error("Error during model streaming: %s", exc.error)
        return None


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------
def prompt_template(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """
    Build the final prompt including context and conversation history.

    Args:
        query (str): The latest user query.
        context (str): Search context text to guide the answer.
        history (List[Dict[str, str]]): Prior conversation messages.

    Returns:
        str: Assembled prompt text.
    """
    base_prompt = "You are a knowledgeable chatbot assistant. "

    if context:
        base_prompt += (
            "Use the following context to respond:\nContext:\n"
            + context
            + "\n"
        )
    else:
        base_prompt += "Answer to the best of your knowledge.\n"

    if history:
        base_prompt += "Conversation History:\n"
        for message in history:
            speaker = "User" if message["role"] == "user" else "Assistant"
            base_prompt += f"{speaker}: {message['content']}\n"
        base_prompt += "\n"

    final_prompt = f"{base_prompt}User: {query}\nAssistant:"
    log.info("Prompt successfully constructed with context and history.")
    return final_prompt


# ---------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------
def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable[str]]:
    """
    Generate a chatbot response using hybrid search context and chat history.

    Args:
        query (str): The user question.
        use_hybrid_search (bool): Whether to include hybrid search results.
        num_results (int): Number of retrieved docs to include as context.
        temperature (float): Response temperature setting.
        chat_history (Optional[List[Dict[str, str]]]): Past conversation history.

    Returns:
        Optional[Iterable[str]]: Stream of response text chunks.
    """
    chat_history = chat_history or []
    recent_history = chat_history[-10:]  # Limit context to last 10 messages
    context_snippets = ""

    # Fetch hybrid search results if enabled
    if use_hybrid_search:
        log.info("Performing hybrid search...")
        search_query = f"passage: {query}" if ASSYMETRIC_EMBEDDING else query

        embedder = get_embedding_model()
        vector = embedder.encode(search_query).tolist()

        results = hybrid_search(query, vector, top_k=num_results)
        log.info("Hybrid search completed with %d results.", len(results))

        for idx, item in enumerate(results):
            context_snippets += f"Document {idx}:\n{item['_source']['text']}\n\n"

    # Construct the final prompt
    prompt = prompt_template(query, context_snippets, recent_history)

    # Run model with streaming
    return run_llama_streaming(prompt, temperature)
