"""
Streamlit page for the Gen AI Chatbot.
Handles UI layout, sidebar controls, model loading, and response streaming.
"""

import logging
import os

import streamlit as st

from src.chat import (  # type: ignore
    ensure_model_pulled,
    generate_response_streaming,
    get_embedding_model,
)
from src.ingestion import create_index, get_opensearch_client
from src.constants import OLLAMA_MODEL_NAME, OPENSEARCH_INDEX
from src.utils import setup_logging


# ---------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="Gen AI - Chatbot", page_icon="🤖")

# Inject custom CSS styles
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
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
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
    h1, h2, h3, h4 { color: #006d77; }
    .stChatMessage {
        background-color: #e0f7fa;
        color: #006d77;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #118ab2;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
logger.info("Custom CSS successfully applied.")


# ---------------------------------------------------------------------
# Chatbot rendering
# ---------------------------------------------------------------------
def render_chatbot_page() -> None:
    """
    Renders the chatbot interface:
    - Sidebar controls (RAG toggle, results count, temperature)
    - Chat history display
    - Handles user prompts and response streaming
    """
    st.title("Gen AI - Chatbot 🤖")

    # Placeholder for model loading status
    model_status = st.empty()

    # Session state defaults
    st.session_state.setdefault("use_hybrid_search", True)
    st.session_state.setdefault("num_results", 5)
    st.session_state.setdefault("temperature", 0.7)

    # Connect to OpenSearch
    with st.spinner("Connecting to OpenSearch..."):
        client = get_opensearch_client()
    create_index(client)

    # Sidebar controls
    st.session_state["use_hybrid_search"] = st.sidebar.checkbox(
        "Enable RAG mode", value=st.session_state["use_hybrid_search"]
    )
    st.session_state["num_results"] = st.sidebar.number_input(
        "Number of Results in Context Window",
        min_value=1,
        max_value=10,
        value=st.session_state["num_results"],
        step=1,
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )

    # Sidebar branding/footer
    st.sidebar.markdown("<h2 style='text-align: center;'>Gen AI</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Your Conversational Platform</h4>",
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
    logger.info("Sidebar configured with controls and branding.")

    # Display spinner while loading models
    with model_status.container():
        st.spinner("Loading models...")

    # Load embedding + Ollama models if not already cached
    if "embedding_models_loaded" not in st.session_state:
        with model_status:
            with st.spinner("Initializing embedding + Ollama models..."):
                get_embedding_model()
                ensure_model_pulled(OLLAMA_MODEL_NAME)
                st.session_state["embedding_models_loaded"] = True
        model_status.empty()
        logger.info("Embedding and Ollama models initialized.")

    # Initialize chat history
    st.session_state.setdefault("chat_history", [])

    # Render chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        logger.info("User query captured.")

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                assembled_text = ""

                response_stream = generate_response_streaming(
                    prompt,
                    use_hybrid_search=st.session_state["use_hybrid_search"],
                    num_results=st.session_state["num_results"],
                    temperature=st.session_state["temperature"],
                    chat_history=st.session_state["chat_history"],
                )

            # Stream tokens into UI
            if response_stream is not None:
                for chunk in response_stream:
                    if (
                        isinstance(chunk, dict)
                        and "message" in chunk
                        and "content" in chunk["message"]
                    ):
                        assembled_text += chunk["message"]["content"]
                        response_placeholder.markdown(assembled_text + "▌")
                    else:
                        logger.error("Unexpected response chunk format.")

            response_placeholder.markdown(assembled_text)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": assembled_text}
            )
            logger.info("Assistant response streamed and displayed.")


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    render_chatbot_page()
