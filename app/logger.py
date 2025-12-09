import logging
import sys


def get_logger(name: str = "rag_chatbot"):
    """Create or return a configured logger instance.

    Args:
        name: Logger name to fetch or create.

    Returns:
        logging.Logger: Logger configured for stdout with a simple formatter.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.propagate = False

    return logger
