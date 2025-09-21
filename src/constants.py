"""
Global configuration constants for the Gen AI project.

This file centralizes model, embedding, logging, and OpenSearch settings
so they can be reused consistently across the application.
"""

# ---------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------

# Path or identifier of the embedding model.
# Examples:
#   - Local path: "embedding_model/"
#   - Hugging Face hub model: "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_PATH: str = "sentence-transformers/all-mpnet-base-v2"

# Whether to use asymmetric embeddings (True) or not (False).
ASSYMETRIC_EMBEDDING: bool = False

# Dimensionality of the embedding model
EMBEDDING_DIMENSION: int = 768

# Maximum characters per text chunk when splitting documents
TEXT_CHUNK_SIZE: int = 300


# ---------------------------------------------------------------------
# Ollama model
# ---------------------------------------------------------------------
# Name of the model used by Ollama for chatbot responses
OLLAMA_MODEL_NAME: str = "llama3.2:1b"


# ---------------------------------------------------------------------
# Fixed application settings (do not change)
# ---------------------------------------------------------------------

# Logging
LOG_FILE_PATH: str = "logs/app.log"  # Output file path for application logs

# OpenSearch configuration
OPENSEARCH_HOST: str = "localhost"   # OpenSearch hostname
OPENSEARCH_PORT: int = 9200          # OpenSearch port
OPENSEARCH_INDEX: str = "documents"  # Default index for storing documents
