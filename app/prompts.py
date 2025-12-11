from langchain_core.prompts import PromptTemplate

INSTRUCTIONS = """\
You are a helpful expert for mathematical research data and software packages.
Your task is to answer the user's question strictly using the provided context.

Rules:
1. Use ONLY information found in the context. If context is missing, incomplete, or irrelevant, say: 
   "I don't know based on the provided context."
2. Do NOT guess, infer unknown facts, or bring in outside knowledge.
3. Provide a **thorough and well-developed answer**:
   - Explain concepts clearly.
   - Include examples, comparisons, definitions, or short step-by-step guidance when useful.
   - Expand on each relevant point rather than giving minimal bullet lists.
4. Use Markdown for headings, lists, and code blocks.
5. When helpful, provide short actionable steps or minimal code examples.
6. If multiple context chunks disagree, state the conflict rather than choosing one.
7. When citing, reference the chunk titles/pages if available (e.g., “According to Page 3…”).
8. Never mention that you are following rules or a prompt.

Your answers must be **substantive, detailed, and educational**, even when the context is short.
Make the answer feel complete, not abbreviated.
"""

QA_PROMPT = PromptTemplate(
    template="""
<instructions>
{instructions}
</instructions>

<context>
The following context chunks were retrieved from the vector database. 
Use them as the ONLY source of truth.

NOTE:
- Context chunks may be partial, fragmented, or out of order.
- They may come from different sections or documents.
- If a detail is unclear, missing, or unsupported, say so explicitly.
- When summarizing context, you must elaborate logically and explain the meaning of the information provided.
- If you refer to the context, say that your knowledge comes from the MaRDI knowledge graph. 

{context}
</context>

{history_block}

<question>
{question}
</question>

Write a **complete and detailed answer**, not a short summary.

Answer:
""",
    input_variables=["instructions", "context", "question", "history_block"],
)

def history_to_text(chat_history):
    """Convert chat history into a compact text block."""
    return "\n".join(f"[{m['role']}]: {m['content']}" for m in chat_history)

def build_prompt(question: str, context: str, history=None) -> str:
    """Build the full prompt string with instructions, context, and optional history."""
    history_block = ""
    if history:
        history_block = f"<history>\n{history_to_text(history)}\n</history>"

    return QA_PROMPT.format(
        instructions=INSTRUCTIONS.strip(),
        context=context or "No context retrieved.",
        question=question,
        history_block=history_block,
    )