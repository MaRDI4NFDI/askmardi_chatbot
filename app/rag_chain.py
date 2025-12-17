import time
import streamlit as st
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama

from app.logger import get_logger
from app.config import load_config
from app.retriever import create_retriever, format_docs
from app.embedder_tools import EmbedderTools
from app.mardi_wiki_retriever import MardiWikiRetriever


logger = get_logger("rag_chain")


@st.cache_resource
def build_cached_chain(
    qdrant_url,
    qdrant_api_key,
    collection,
    retrieve_limit,
    context_limit,
    candidate_multiplier,
    embed_model,
    llm_host,
    llm_model,
    ollama_api_key,
    llm_context_size=None,
    _reranker=None,
):
    """Build and cache the retrieval chain and LLM client to avoid per-prompt rebuilds."""

    t_start = time.time()

    # --------------------------------------------------
    # Qdrant client
    # --------------------------------------------------
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    logger.info("Initialized Qdrant client (cached)")

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    t_embed = time.time()
    embedder = EmbedderTools(model_name=embed_model)
    embeddings = embedder.embeddings
    logger.info("Loaded embeddings in %.2fs (cached)", time.time() - t_embed)

    # --------------------------------------------------
    # MaRDI Wiki retriever (cached)
    # --------------------------------------------------
    mardi_wiki = MardiWikiRetriever(
        api_url="https://portal.mardi4nfdi.de/w/api.php",
        top_k=retrieve_limit,
    )
    logger.info("Initialized MaRDI Wiki retriever (cached)")

    # --------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------
    retriever_fn = create_retriever(
        client=client,
        embeddings=embeddings,
        collection=collection,
        limit=retrieve_limit,
        candidate_multiplier=candidate_multiplier,
        reranker=_reranker,
        mardi_wiki=mardi_wiki,
    )

    def timed_retriever(x):
        """Retrieve documents while tracking duration for logging.

        Args:
            x: Either a query string or a dict containing:
               - question (str)
               - progress_cb (callable | None)

        Returns:
            list[Document]: Retrieved documents.
        """
        if isinstance(x, dict):
            query = x["question"]
            progress_cb = x.get("progress_cb")
        else:
            query = x
            progress_cb = None

        t0 = time.time()
        try:
            docs = retriever_fn(query, progress_cb=progress_cb)
        except Exception:
            logger.exception("Retriever failed")
            raise

        logger.info("Retrieval: %.2fs", time.time() - t0)
        return docs


    retriever = RunnableLambda(timed_retriever)

    # --------------------------------------------------
    # LLM client
    # --------------------------------------------------
    llm = ChatOllama(
        base_url=llm_host,
        model=llm_model,
        temperature=0,
        streaming=True,
        client_kwargs=(
            {"headers": {"Authorization": f"Bearer {ollama_api_key}"}}
            if ollama_api_key
            else None
        ),
        num_ctx=llm_context_size,
    )
    logger.info("LLM client ready (model=%s)", llm_model)

    # --------------------------------------------------
    # Chain assembly
    # --------------------------------------------------
    def assemble(x):
        """Format retriever output into docs + string context payload.

        Args:
            x (dict): Input dict containing ``docs`` from the retriever.

        Returns:
            dict: Dict with ``docs`` passthrough and concatenated ``context`` string.
        """
        docs = x["docs"]
        return {
            "docs": docs,
            "context": format_docs(docs, context_limit),
        }

    chain = (
            RunnablePassthrough()
            | {
                "docs": lambda x: retriever.invoke(
                    {
                        "question": x["question"],
                        "progress_cb": x.get("progress_cb"),
                    }
                ),
            }
            | RunnableLambda(assemble)
    )

    logger.info("Chain built in %.2fs (cached)", time.time() - t_start)
    return chain, llm


def build_rag_chain(
    collection_override: str | None = None,
    reranker=None,
):
    """Build the retrieval chain and LLM client."""

    t_start = time.time()
    cfg = load_config()
    logger.info("Config loaded in %.2fs", time.time() - t_start)

    collection_name = collection_override or cfg["qdrant"]["collection"]
    if collection_override:
        logger.info("Using Qdrant collection override: %s", collection_name)
    else:
        logger.info("Using default Qdrant collection from config: %s", collection_name)

    chain, llm = build_cached_chain(
        qdrant_url=cfg["qdrant"]["url"],
        qdrant_api_key=cfg["qdrant"].get("api_key"),
        collection=collection_name,
        retrieve_limit=cfg["qdrant"].get("limit", 10),
        context_limit=cfg["qdrant"].get("context_limit", 10),
        candidate_multiplier=cfg["qdrant"].get("candidate_multiplier", 4),
        embed_model=cfg["embedding"]["model_name"],
        llm_host=cfg["ollama"]["host"],
        llm_model=cfg["ollama"]["model_name"],
        ollama_api_key=cfg["ollama"].get("api_key"),
        llm_context_size=cfg["ollama"].get("max_context_size"),
        _reranker=reranker,
    )

    return chain, llm
