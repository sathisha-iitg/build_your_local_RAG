"""
OpenSearch client utilities and hybrid search functionality.
"""

import logging
from typing import Any, Dict, List

from opensearchpy import OpenSearch

from src.constants import OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_PORT
from src.utils import setup_logging

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------
def get_opensearch_client() -> OpenSearch:
    """
    Create and return an OpenSearch client instance.

    Returns:
        OpenSearch: Configured client object.
    """
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    log.info("OpenSearch client initialized and ready.")
    return client


# ---------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------
def hybrid_search(
    query_text: str, query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Run a hybrid search (text + vector) against the OpenSearch index.

    Args:
        query_text (str): Text query for full-text search.
        query_embedding (List[float]): Embedding vector for ANN search.
        top_k (int, optional): Number of results to retrieve. Default is 5.

    Returns:
        List[Dict[str, Any]]: Search results as a list of documents.
    """
    client = get_opensearch_client()

    body = {
        "_source": {"exclude": ["embedding"]},  # Exclude embeddings from response
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text": {"query": query_text}}},
                    {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                ]
            }
        },
        "size": top_k,
    }

    response = client.search(
        index=OPENSEARCH_INDEX, body=body, search_pipeline="nlp-search-pipeline"
    )
    log.info("Hybrid search executed for query='%s' with top_k=%d.", query_text, top_k)

    results: List[Dict[str, Any]] = response["hits"]["hits"]
    return results
