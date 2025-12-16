import time
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client.http.exceptions import ResponseHandlingException
from flashrank import RerankRequest

from app.logger import get_logger

logger = get_logger("retriever")


# ---------------------------------------------------------------------
# Reranking helper (FlashRank)
# ---------------------------------------------------------------------

def rerank_docs(
    query: str,
    docs: List[Document],
    top_k: int,
    ranker,
) -> List[Document]:
    """Rerank documents using a provided FlashRank ranker.

    Args:
        query: User query.
        docs: Candidate documents from vector search.
        top_k: Number of documents to keep after reranking.
        ranker: Initialized FlashRank Ranker instance.

    Returns:
        List[Document]: Reranked documents.
    """
    if not docs:
        return docs

    passages = [
        {
            "id": str(i),
            "text": doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]

    request = RerankRequest(
        query=query,
        passages=passages,
    )

    reranked = ranker.rerank(request)

    ranked_docs: List[Document] = []
    for r in reranked[:top_k]:
        doc = docs[int(r["id"])]
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["rerank_score"] = r["score"]
        ranked_docs.append(doc)

    # --------------------------------------------------
    # BEFORE: embedding-based ranking (Qdrant order)
    # --------------------------------------------------
    logger.info("Before rerank (Qdrant top %d):", top_k)
    for i, doc in enumerate(docs[:top_k], start=1):
        meta = doc.metadata or {}
        logger.info(
            "  #%02d src=%s page=%s",
            i,
            meta.get("title", "unknown"),
            meta.get("page", "?"),
        )

    # --------------------------------------------------
    # AFTER: cross-encoder reranking
    # --------------------------------------------------
    logger.info("After rerank (FlashRank top %d):", top_k)
    for i, r in enumerate(reranked[:top_k], start=1):
        doc = docs[int(r["id"])]
        meta = doc.metadata or {}
        logger.info(
            "  #%02d score=%.3f src=%s page=%s",
            i,
            r["score"],
            meta.get("title", "unknown"),
            meta.get("page", "?"),
        )

    return ranked_docs


# ---------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------

def create_retriever(
    client,
    embeddings,
    collection: str,
    limit: int = 5,
    candidate_multiplier: int = 4,
    reranker: Optional[object] = None,
):
    """Create a retriever callable with optional FlashRank reranking.

    Args:
        client: Qdrant client instance.
        embeddings: Embedding model exposing embed_query().
        collection: Target Qdrant collection name.
        limit: Final number of documents to return.
        candidate_multiplier: How many more docs to fetch before reranking.
        reranker: Optional FlashRank Ranker instance.

    Returns:
        Callable[[str], List[Document]]
    """

    def custom_retrieve(query: str) -> List[Document]:
        query_embedding = embeddings.embed_query(query)

        last_exc = None
        res = None

        for attempt in range(1, 6):
            try:
                res = client.query_points(
                    collection_name=collection,
                    query=query_embedding,
                    limit=limit * candidate_multiplier,
                    with_vectors=False,
                )
                break
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

        # --------------------------------------------------------------
        # Optional FlashRank reranking
        # --------------------------------------------------------------
        if reranker is not None:
            logger.info("Reranking %d docs with FlashRank", len(docs))
            docs = rerank_docs(
                query=query,
                docs=docs,
                top_k=limit,
                ranker=reranker,
            )
        else:
            docs = docs[:limit]

        return docs

    return custom_retrieve


# ---------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------

def format_docs(docs: List[Document], context_limit: int) -> str:
    """Join the top documents into a context string for the LLM with provenance.

    Args:
        docs: Documents returned from the retriever.
        context_limit: Max number of docs to include in the context.

    Returns:
        str: Concatenated page contents annotated with metadata.
    """
    logger.info("Formatting %d docs for LLM context", len(docs))
    formatted = []
    for doc in docs[:context_limit]:
        meta = doc.metadata or {}
        package = meta.get("package", "unknown")
        version = meta.get("version", "unknown")
        page = meta.get("page", "unknown")
        chunk_idx = meta.get("chunk_index", "unknown")
        rerank_score = meta.get("rerank_score", "n/a")

        prefix = (
            f"[{package} v{version} — page {page}, "
            f"chunk {chunk_idx}, score {rerank_score}]"
        )
        formatted.append(f"{prefix}\n{doc.page_content}")

    return "\n\n".join(formatted)
