import re
from app.config import load_config
from app.logger import get_logger

logger = get_logger("ui")


def inspect_usage_line(log_line):
    """Extract token counts from a usage log line and warn if input crosses threshold.

    Args:
        log_line: Raw log line containing token usage details.

    Returns:
        tuple | None: Parsed (input_tokens, output_tokens, total_tokens) if found, else None.
    """
    cfg = load_config()
    threshold = cfg["ollama"].get("max_context_size")

    match = re.search(
        r"input_tokens':\s*(\d+).*output_tokens':\s*(\d+).*total_tokens':\s*(\d+)",
        log_line,
    )
    if not match:
        return None
    input_tokens, output_tokens, total_tokens = map(int, match.groups())
    if threshold and input_tokens >= threshold:
        logger.warning(
            "LLM input tokens (%d) exceeded threshold (%d)",
            input_tokens,
            threshold,
        )
    else:
        logger.debug(
            "LLM input tokens (%d) within threshold (%d)",
            input_tokens,
            threshold or -1,
        )
    return input_tokens, output_tokens, total_tokens


def estimate_token_count(text: str) -> int:
    """Return a quick token estimate using a mixed heuristic.

    Args:
        text: Input string to approximate token count for.

    Returns:
        int: Approximate token count.
    """
    rough_bpe = max(1, len(text) // 4)
    wordish = len(re.findall(r"\w+|[^\w\s]", text))
    return max(rough_bpe, wordish)


def ns_to_seconds(ns_value):
    """Convert nanoseconds to seconds if present.

    Args:
        ns_value: Duration in nanoseconds or None.

    Returns:
        float | None: Duration in seconds if convertible, else None.
    """
    return ns_value / 1_000_000_000 if ns_value is not None else None


def format_duration(seconds):
    """Format a duration in seconds for logging.

    Args:
        seconds: Duration in seconds or None.

    Returns:
        str: Formatted string like ``0.12s`` or ``n/a`` if missing.
    """
    return f"{seconds:.2f}s" if seconds is not None else "n/a"
