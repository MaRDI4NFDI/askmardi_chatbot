import time
import streamlit as st
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from app.logger import get_logger
from app.config import load_config
from app.retriever import create_retriever, format_docs
from app.embedder_tools import EmbedderTools


logger = get_logger("rag_chain")


@st.cache_resource
def build_cached_chain(qdrant_url, qdrant_api_key, collection, embed_model, llm_host, llm_model, llm_api_key):
    """Build and cache the retrieval chain and LLM client to avoid per-prompt rebuilds.

    Args:
        qdrant_url: Qdrant endpoint URL.
        qdrant_api_key: Optional Qdrant API key.
        collection: Target collection name.
        embed_model: Embedding model name.
        llm_host: OpenAI-compatible host URL.
        llm_model: LLM model name.
        llm_api_key: Optional LLM API key.

    Returns:
        Tuple[Runnable, ChatOpenAI]: Chain plus LLM client for streaming.
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

    llm = ChatOpenAI(
        base_url=llm_host,
        api_key=llm_api_key,
        model_name=llm_model,
        temperature=0,
        streaming=True,
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
            "context": format_docs(docs),
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
        Tuple[Runnable, ChatOpenAI]: Chain that fetches docs and context, plus the LLM client for streaming.
    """
    t_start = time.time()
    cfg = load_config()
    logger.info("Config loaded in %.2fs", time.time() - t_start)

    chain, llm = build_cached_chain(
        qdrant_url=cfg["qdrant"]["url"],
        qdrant_api_key=cfg["qdrant"].get("api_key"),
        collection=cfg["qdrant"]["collection"],
        embed_model=cfg["embedding"]["model_name"],
        llm_host=cfg["llm"]["host"],
        llm_model=cfg["llm"]["model_name"],
        llm_api_key=cfg["llm"].get("api_key"),
    )

    return chain, llm
