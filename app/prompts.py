from langchain_core.prompts import PromptTemplate

INSTRUCTIONS = """\
You are a helpful RAG assistant. Use only the provided context chunks to answer.
- Do not make up facts; say "I don't know." when the answer is not present.
- Keep answers concise and beginner-friendly; use Markdown for structure.
- If relevant, mention source titles/pages inline.
- Prefer actionable steps and short code snippets when useful.
"""

QA_PROMPT = PromptTemplate(
    template="""<instructions>
{instructions}
</instructions>
<context>
{context}
</context>
{history_block}
<question>
{question}
</question>
Answer:
""",
    input_variables=["instructions", "context", "question", "history_block"],
)


def history_to_text(chat_history):
    """Convert chat history into a compact text block.

    Args:
        chat_history: Sequence of message dicts containing role and content.

    Returns:
        str: Multi-line string with roles prepended to message content.
    """
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in chat_history)


def build_prompt(question: str, context: str, history=None) -> str:
    """Build the full prompt string with instructions, context, and optional history.

    Args:
        question: User's question to answer.
        context: Retrieved context string from Qdrant.
        history: Optional chat history list to include for continuity.

    Returns:
        str: Fully formatted prompt ready for the LLM.
    """
    history_block = ""
    if history:
        history_block = f"<history>\n{history_to_text(history)}\n</history>"

    return QA_PROMPT.format(
        instructions=INSTRUCTIONS.strip(),
        context=context or "No context retrieved.",
        question=question,
        history_block=history_block,
    )
