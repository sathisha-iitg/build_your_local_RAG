"""
Document ingestion utilities for OpenSearch.

This module provides helpers to create, delete, and populate indices
with documents and their embeddings for semantic search.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from opensearchpy import OpenSearch, helpers

from src.constants import ASSYMETRIC_EMBEDDING, EMBEDDING_DIMENSION, OPENSEARCH_INDEX
from src.opensearch import get_opensearch_client
from src.utils import setup_logging

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Index configuration and lifecycle
# ---------------------------------------------------------------------
def load_index_config() -> Dict[str, Any]:
    """
    Load index configuration from JSON file and adjust for embedding dimension.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open("src/index_config.json", "r") as config_file:
        config = json.load(config_file)

    # Patch embedding dimension dynamically
    config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
    log.info("Index configuration loaded from src/index_config.json.")
    return config if isinstance(config, dict) else {}


def create_index(client: OpenSearch) -> None:
    """
    Create a new index if it does not already exist.

    Args:
        client (OpenSearch): Client instance.
    """
    body = load_index_config()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        resp = client.indices.create(index=OPENSEARCH_INDEX, body=body)
        log.info("Index '%s' created: %s", OPENSEARCH_INDEX, resp)
    else:
        log.info("Index '%s' already exists.", OPENSEARCH_INDEX)


def delete_index(client: OpenSearch) -> None:
    """
    Delete the index if it is present.

    Args:
        client (OpenSearch): Client instance.
    """
    if client.indices.exists(index=OPENSEARCH_INDEX):
        resp = client.indices.delete(index=OPENSEARCH_INDEX)
        log.info("Index '%s' deleted: %s", OPENSEARCH_INDEX, resp)
    else:
        log.info("Index '%s' does not exist.", OPENSEARCH_INDEX)


# ---------------------------------------------------------------------
# Document operations
# ---------------------------------------------------------------------
def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Index multiple documents in bulk.

    Args:
        documents (List[Dict[str, Any]]): Each item must include
            - doc_id
            - text
            - embedding
            - document_name

    Returns:
        Tuple[int, List[Any]]: (Number indexed, list of errors)
    """
    client = get_opensearch_client()
    actions = []

    for doc in documents:
        doc_id = doc["doc_id"]
        embedding_vector = doc["embedding"].tolist()
        doc_name = doc["document_name"]

        # Prefix text for asymmetric embedding, if enabled
        text_value = f"passage: {doc['text']}" if ASSYMETRIC_EMBEDDING else doc["text"]

        actions.append(
            {
                "_index": OPENSEARCH_INDEX,
                "_id": doc_id,
                "_source": {
                    "text": text_value,
                    "embedding": embedding_vector,
                    "document_name": doc_name,
                },
            }
        )

    # Execute bulk operation
    success_count, error_list = helpers.bulk(client, actions)
    log.info(
        "Bulk indexed %d documents into '%s' with %d errors.",
        len(documents),
        OPENSEARCH_INDEX,
        len(error_list),
    )
    return success_count, error_list


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Remove documents by their `document_name` field.

    Args:
        document_name (str): Target document name.

    Returns:
        Dict[str, Any]: Response payload from delete-by-query.
    """
    client = get_opensearch_client()
    query = {"query": {"term": {"document_name": document_name}}}

    resp: Dict[str, Any] = client.delete_by_query(index=OPENSEARCH_INDEX, body=query)
    log.info(
        "Deleted documents where document_name='%s' from index '%s'.",
        document_name,
        OPENSEARCH_INDEX,
    )
    return resp
