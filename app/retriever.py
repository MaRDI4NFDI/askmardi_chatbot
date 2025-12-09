from typing import List
from langchain_core.documents import Document


def create_retriever(client, embeddings, collection: str):
    """Create a retriever callable that queries Qdrant with embeddings.

    Args:
        client: Qdrant client instance.
        embeddings: Embedding model exposing embed_query.
        collection: Target Qdrant collection name.

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
        res = client.query_points(
            collection_name=collection,
            with_vectors=True,
            query=query_embedding,
            limit=5,
        )
        docs = [
            Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload
            )
            for p in res.points
        ]
        return docs

    return custom_retrieve


def format_docs(docs: List[Document]) -> str:
    """Join the top documents into a context string for the LLM.

    Args:
        docs: Documents returned from the retriever.

    Returns:
        str: Concatenated page contents from the first few documents.
    """
    return "\n\n".join(doc.page_content for doc in docs[:3])
