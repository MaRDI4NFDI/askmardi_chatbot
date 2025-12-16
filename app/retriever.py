import time
from typing import List
from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException
from app.logger import get_logger

logger = get_logger("retriever")


def create_retriever(client, embeddings, collection: str, limit: int = 5):
    """Create a retriever callable that queries Qdrant with embeddings.

    Args:
        client: Qdrant client instance.
        embeddings: Embedding model exposing embed_query.
        collection: Target Qdrant collection name.
        limit: Maximum number of documents to retrieve from Qdrant.

    Returns:
        Callable[[str], List[Document]]: Function that performs similarity search.
    """

    def custom_retrieve(query: str) -> List[Document]:
        """Retrieve the top documents for the given query from Qdrant.

        Args:
            query: Natural language query string.

        Returns:
            List[Document]: Retrieved documents with payload metadata.
        """
        query_embedding = embeddings.embed_query(query)

        last_exc = None
        res = None

        for attempt in range(1, 6):
            try:
                res = client.query_points(
                    collection_name=collection,
                    with_vectors=True,
                    query=query_embedding,
                    limit=limit,
                )
                break  # success
            except ResponseHandlingException as exc:
                last_exc = exc
                logger.warning(
                    "Qdrant query failed (attempt %d/5): %s",
                    attempt,
                    exc,
                )
                if attempt < 5:
                    time.sleep(1)

        if res is None:
            logger.error(
                "Qdrant query failed after 5 attempts. Raising last exception."
            )
            raise last_exc

        docs = [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload,
            )
            for p in res.points
        ]
        return docs

    return custom_retrieve


def format_docs(docs: List[Document], context_limit: int) -> str:
    """Join the top documents into a context string for the LLM with provenance.

    Args:
        docs: Documents returned from the retriever.
        context_limit: Max number of docs to include in the context.

    Returns:
        str: Concatenated page contents from the first few documents, annotated with metadata.
    """
    formatted = []
    for doc in docs[:context_limit]:
        meta = doc.metadata or {}
        package = meta.get("package", "unknown")
        version = meta.get("version", "unknown")
        page = meta.get("page", "unknown")
        chunk_idx = meta.get("chunk_index", "unknown")
        prefix = f"[{package} v{version} — page {page}, chunk {chunk_idx}]"
        formatted.append(f"{prefix}\n{doc.page_content}")
    return "\n\n".join(formatted)
