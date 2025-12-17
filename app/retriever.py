import time
from collections import Counter
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException
from flashrank import RerankRequest

from app.logger import get_logger

logger = get_logger("retriever")


# ---------------------------------------------------------------------
# Query heuristics
# ---------------------------------------------------------------------

def is_lexical_query(query: str) -> bool:
    """Heuristic for identifier-style or short lexical queries."""
    q = query.strip()
    return (
        len(q) <= 6
        or q.isupper()
        or any(c.isdigit() for c in q)
    )


# ---------------------------------------------------------------------
# Reranking helper (FlashRank)
# ---------------------------------------------------------------------

def rerank_docs(
    query: str,
    docs: List[Document],
    top_k: int,
    ranker,
) -> List[Document]:
    """Rerank documents using a provided FlashRank ranker."""
    if not docs:
        return docs

    passages = [
        {"id": str(i), "text": doc.page_content}
        for i, doc in enumerate(docs)
    ]

    request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(request)

    ranked_docs: List[Document] = []
    for r in reranked[:top_k]:
        doc = docs[int(r["id"])]
        doc.metadata = dict(doc.metadata or {})

        # --------------------------------------------------
        # Sparse bias (small, controlled)
        # --------------------------------------------------
        bias = doc.metadata.get("rerank_bias", 0.0)
        doc.metadata["rerank_score"] = r["score"] + bias

        ranked_docs.append(doc)

    # --------------------------------------------------
    # BEFORE: candidate composition
    # --------------------------------------------------
    before_counts = Counter(
        d.metadata.get("retriever", "unknown") for d in docs
    )
    logger.info(
        "Before rerank: %d candidates | dense=%d sparse=%d",
        len(docs),
        before_counts.get("dense", 0),
        before_counts.get("sparse", 0),
    )

    # --------------------------------------------------
    # AFTER: final composition
    # --------------------------------------------------
    after_counts = Counter(
        d.metadata.get("retriever", "unknown") for d in ranked_docs
    )
    logger.info(
        "After rerank: top-%d | dense=%d sparse=%d",
        top_k,
        after_counts.get("dense", 0),
        after_counts.get("sparse", 0),
    )

    # --------------------------------------------------
    # Detailed per-document trace (DEBUG)
    # --------------------------------------------------
    for i, doc in enumerate(ranked_docs, start=1):
        meta = doc.metadata or {}
        logger.debug(
            "Final #%02d retriever=%s dense_score=%s sparse_score=%s "
            "rerank_score=%.4f bias=%.2f",
            i,
            meta.get("retriever"),
            meta.get("dense_score"),
            meta.get("sparse_score"),
            meta.get("rerank_score"),
            meta.get("rerank_bias", 0.0),
        )

    return ranked_docs


# ---------------------------------------------------------------------
# Sparse retriever
# ---------------------------------------------------------------------

def sparse_retrieve(
    client,
    collection: str,
    query: str,
    limit: int,
) -> List[Document]:
    """Retrieve documents using Qdrant full-text (BM25-like) search."""

    res = client.query_points(
        collection_name=collection,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="page_content",
                    match=models.MatchText(text=query),
                )
            ]
        ),
        limit=limit,
        with_vectors=False,
    )

    docs: List[Document] = []
    for p in res.points:
        meta = dict(p.payload or {})
        meta["retriever"] = "sparse"
        meta["sparse_score"] = p.score

        docs.append(
            Document(
                page_content=meta.pop("page_content", ""),
                metadata=meta,
            )
        )

    return docs


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
    """Create a hybrid retriever using dense + sparse Qdrant search."""

    logger.info(f"[create_retriever] limit={limit}; candidate_multiplier={candidate_multiplier} ")

    def custom_retrieve(query: str) -> List[Document]:
        query_embedding = embeddings.embed_query(query)
        dense_res = None
        last_exc = None

        # --------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------
        for attempt in range(1, 6):
            try:
                dense_res = client.query_points(
                    collection_name=collection,
                    query=query_embedding,
                    limit=limit * candidate_multiplier,
                    with_vectors=False,
                )
                break
            except ResponseHandlingException as exc:
                last_exc = exc
                logger.warning(
                    "Qdrant dense query failed (attempt %d/5): %s",
                    attempt,
                    exc,
                )
                if attempt < 5:
                    time.sleep(1)

        if dense_res is None:
            logger.error("Dense Qdrant query failed after 5 attempts.")
            raise last_exc

        dense_docs: List[Document] = []
        for p in dense_res.points:
            meta = dict(p.payload or {})
            meta["retriever"] = "dense"
            meta["dense_score"] = p.score

            dense_docs.append(
                Document(
                    page_content=meta.pop("page_content", ""),
                    metadata=meta,
                )
            )

        # --------------------------------------------------
        # Sparse retrieval (query-dependent routing)
        # --------------------------------------------------
        sparse_limit = limit * candidate_multiplier
        if is_lexical_query(query):
            sparse_limit *= 2
            logger.debug(
                "Lexical query detected → boosting sparse retrieval (limit=%d)",
                sparse_limit,
            )

        sparse_docs = sparse_retrieve(
            client=client,
            collection=collection,
            query=query,
            limit=sparse_limit,
        )

        logger.info(
            "Retrieved candidates | dense=%d sparse=%d",
            len(dense_docs),
            len(sparse_docs),
        )

        # --------------------------------------------------
        # Merge & deduplicate
        # --------------------------------------------------
        seen = set()
        merged: List[Document] = []

        def doc_key(doc: Document) -> str:
            m = doc.metadata or {}
            return (
                f"{m.get('qid','?')}:"
                f"{m.get('component','?')}:"
                f"{m.get('chunk_index','?')}"
            )

        for doc in dense_docs + sparse_docs:
            key = doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        logger.info("Merged candidates after deduplication: %d", len(merged))

        # --------------------------------------------------
        # Sparse bias before reranking
        # --------------------------------------------------
        for doc in merged:
            meta = doc.metadata or {}
            meta["rerank_bias"] = 0.1 if meta.get("retriever") == "sparse" else 0.0

        # --------------------------------------------------
        # Optional FlashRank reranking
        # --------------------------------------------------
        if reranker is not None:
            logger.info("Reranking %d merged docs with FlashRank", len(merged))
            merged = rerank_docs(
                query=query,
                docs=merged,
                top_k=limit,
                ranker=reranker,
            )
        else:
            merged = merged[:limit]

        return merged

    return custom_retrieve


# ---------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------

def format_docs(docs: List[Document], context_limit: int) -> str:
    """Join the top documents into a context string for the LLM with provenance."""
    logger.info("[format_docs] Formatting %d docs for LLM context (context_limit: %d)", len(docs), context_limit)
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
