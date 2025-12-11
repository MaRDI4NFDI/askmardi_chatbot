import os
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from app.logger import get_logger

logger = get_logger("config")


def load_config():
    """Load configuration from config.yaml with optional env overrides.

    Returns:
        dict: Parsed configuration dictionary with potential LLM API key override.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Allow API keys to be overridden via env
    if "OLLAMA_API_KEY" in os.environ:
        cfg["ollama"]["api_key"] = os.environ["OLLAMA_API_KEY"]
        logger.info("OLLAMA_API_KEY overridden via environment")

    # Allow qdrant url to be overridden via env
    if "QDRANT_URL" in os.environ:
        cfg["qdrant"]["url"] = os.environ["QDRANT_URL"]
        logger.info("QDRANT_URL overridden via environment")

    logger.info("Config loaded (ollama.model_name=%s, qdrant.url=%s)", cfg["ollama"]["model_name"], cfg["qdrant"]["url"])
    return cfg


def check_config():
    """Validate config by loading it and checking Qdrant and LLM reachability.

    Returns:
        bool: True if both checks succeed, False otherwise.
    """
    cfg = load_config()
    ok = True

    try:
        client = QdrantClient(
            url=cfg["qdrant"]["url"],
            api_key=cfg["qdrant"].get("api_key"),
        )
        client.get_collections()
        logger.info("Qdrant connectivity check succeeded")
    except Exception as exc:
        ok = False
        logger.warning("Qdrant connectivity check failed: %s", exc)

    try:
        llm_client = OpenAI(
            base_url=cfg["ollama"]["host"],
            api_key=cfg["ollama"].get("api_key"),
        )
        llm_client.models.list()
        logger.info("LLM connectivity check succeeded")
    except Exception as exc:
        ok = False
        logger.warning("LLM connectivity check failed: %s", exc)

    return ok
