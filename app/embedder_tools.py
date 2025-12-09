from langchain_huggingface import HuggingFaceEmbeddings


class EmbedderTools:
    """Minimal embedding utility for RAG inference."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model.

        Args:
            model_name: Hugging Face embedding model name to load for queries.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_text(self, text: str):
        """Return embedding vector for a given string.

        Args:
            text: Input text to embed.

        Returns:
            List[float]: Embedding vector for the provided text.
        """
        return self.embeddings.embed_query(text)
