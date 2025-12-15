import logging

import app.helper as helper


class ListHandler(logging.Handler):
    """Capture log messages for assertions."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))


def test_inspect_usage_line_warns_at_threshold(monkeypatch, caplog):
    """Warn when input tokens are at or above the configured threshold."""
    monkeypatch.setattr(
        helper,
        "load_config",
    lambda: {"ollama": {"max_context_size": 100}},
    )
    handler = ListHandler()
    handler.setLevel(logging.WARNING)
    helper.logger.addHandler(handler)

    usage_line = "LLM usage (last seen): {'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}"
    parsed = helper.inspect_usage_line(usage_line)

    assert parsed == (100, 10, 110)
    assert any("exceeded threshold" in msg for msg in handler.messages)
    helper.logger.removeHandler(handler)


def test_inspect_usage_line_below_threshold(monkeypatch, caplog):
    """Do not warn when input tokens are below the threshold."""
    monkeypatch.setattr(
        helper,
        "load_config",
        lambda: {"ollama": {"max_context_size": 200}},
    )
    handler = ListHandler()
    handler.setLevel(logging.WARNING)
    helper.logger.addHandler(handler)

    usage_line = "LLM usage (last seen): {'input_tokens': 50, 'output_tokens': 10, 'total_tokens': 60}"
    parsed = helper.inspect_usage_line(usage_line)

    assert parsed == (50, 10, 60)
    assert not any("exceeded threshold" in msg for msg in handler.messages)
    helper.logger.removeHandler(handler)


def test_estimate_token_count():
    """Estimate tokens using heuristic should be non-zero and reasonable."""
    text = "Hello world"
    assert helper.estimate_token_count(text) == 2


def test_time_helpers():
    """Convert nanoseconds to seconds and format durations."""
    assert helper.ns_to_seconds(1_500_000_000) == 1.5
    assert helper.ns_to_seconds(None) is None
    assert helper.format_duration(1.5) == "1.50s"
    assert helper.format_duration(None) == "n/a"
