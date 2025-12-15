import pytest

pytest.importorskip("langchain_core.prompts")
from app import prompts


def test_history_to_text_formats_lines():
    """History lines are joined with role tags."""
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert prompts.history_to_text(history) == "[user]: Hi\n[assistant]: Hello"


def test_build_prompt_includes_sections():
    """Prompt includes instructions, context, question, and history when provided."""
    question = "What is R?"
    context = "R is a language."
    history = [{"role": "user", "content": "Earlier"}]

    built = prompts.build_prompt(question=question, context=context, history=history)

    assert "R is a language." in built
    assert "<question>" in built and question in built
    assert "<history>" in built and "Earlier" in built
    assert "You are a helpful expert" in built
