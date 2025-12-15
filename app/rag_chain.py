import time
import streamlit as st
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama

from app.logger import get_logger
from app.config import load_config
from app.retriever import create_retriever, format_docs
from app.embedder_tools import EmbedderTools


logger = get_logger("rag_chain")


@st.cache_resource
def build_cached_chain(qdrant_url, qdrant_api_key, collection,
                       retrieve_limit, context_limit, embed_model,
                       llm_host, llm_model, ollama_api_key, llm_context_size=None):
    """Build and cache the retrieval chain and LLM client to avoid per-prompt rebuilds.

    Args:
        qdrant_url: Qdrant endpoint URL.
        qdrant_api_key: Optional Qdrant API key.
        collection: Target collection name.
        retrieve_limit: Max number of documents to pull from Qdrant.
        context_limit: Max number of docs to include in the LLM context.
        embed_model: Embedding model name.
        llm_host: Ollama host URL.
        llm_model: LLM model name.
        ollama_api_key: Optional LLM API key.
        llm_context_size: Optional maximum context window for the LLM (tokens).

    Returns:
        Tuple[Runnable, ChatOllama]: Chain plus LLM client for streaming.
    """
    t_start = time.time()

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )
    logger.info("Initialized Qdrant client (cached)")

    t_embed = time.time()
    embedder = EmbedderTools(model_name=embed_model)
    embeddings = embedder.embeddings
    logger.info("Loaded embeddings in %.2fs (cached)", time.time() - t_embed)

    retriever_fn = create_retriever(
        client=client,
        embeddings=embeddings,
        collection=collection,
        limit=retrieve_limit,
    )

    def timed_retriever(query):
        """Wrap the retriever to log latency.

        Args:
            query: User query string.

        Returns:
            list: Retrieved documents.
        """
        t0 = time.time()
        docs = retriever_fn(query)
        logger.info(f"Retrieval: {time.time() - t0:.2f}s")
        return docs

    retriever = RunnableLambda(timed_retriever)

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

    def assemble(x):
        """Package docs and formatted context for downstream consumption.

        Args:
            x: Mapping containing docs.

        Returns:
            dict: Payload with raw docs and formatted context string.
        """
        docs = x["docs"]
        return {
            "docs": docs,
            "context": format_docs(docs, context_limit),
        }

    chain = (
        RunnablePassthrough()
        | {
            "docs": lambda x: retriever.invoke(x["question"]),
        }
        | RunnableLambda(assemble)
    )

    logger.info("Chain built in %.2fs (cached)", time.time() - t_start)
    return chain, llm


def build_rag_chain():
    """Build the retrieval chain and LLM client.

    Returns:
        Tuple[Runnable, ChatOllama]: Chain that fetches docs and context, plus the LLM client for streaming.
    """
    t_start = time.time()
    cfg = load_config()
    logger.info("Config loaded in %.2fs", time.time() - t_start)

    chain, llm = build_cached_chain(
        qdrant_url=cfg["qdrant"]["url"],
        qdrant_api_key=cfg["qdrant"].get("api_key"),
        collection=cfg["qdrant"]["collection"],
        retrieve_limit=cfg["qdrant"].get("limit", 5),
        context_limit=cfg["qdrant"].get("context_limit", 3),
        embed_model=cfg["embedding"]["model_name"],
        llm_host=cfg["ollama"]["host"],
        llm_model=cfg["ollama"]["model_name"],
        ollama_api_key=cfg["ollama"].get("api_key"),
        llm_context_size=cfg["ollama"].get("max_context_size"),
    )

    return chain, llm
