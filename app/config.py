import os
import yaml
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
    if "LLM_API_KEY" in os.environ:
        cfg["llm"]["api_key"] = os.environ["LLM_API_KEY"]
        logger.info("LLM_API_KEY overridden via environment")

    logger.info("Config loaded (llm.model_name=%s, qdrant.url=%s)", cfg["llm"]["model_name"], cfg["qdrant"]["url"])
    return cfg
