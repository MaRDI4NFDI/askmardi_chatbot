import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from app.rag_chain import build_rag_chain
from app.prompts import build_prompt
from app.logger import get_logger


HISTORY_LENGTH = 5
MIN_TIME_BETWEEN_REQUESTS = timedelta(seconds=3)
SUGGESTIONS = {
    "🔔 Timetables": (
        "What information is available about packages that are concerned with timetables?"
    ),
    "🌦️ Weather analysis": (
        "Which packages are available that deal with weather analysis?"
    ),
    "🧮 Math packages": (
        "Which packages are available that have something to do with mathematics?"
    ),
}


# --- Session state ---
def init_state():
    """Initialize all session state keys with defaults if missing.

    Returns:
        None
    """
    defaults = {
        "messages": [],
        "docs": [],
        "queued_message": None,
        "last_prompt": None,
        "prev_question_timestamp": datetime.fromtimestamp(0),
        "chat_input": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
logger = get_logger("ui")


def apply_layout_styles():
    """Constrain page width and add comfortable padding.

    Returns:
        None
    """
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_session():
    """Clear chat, docs, and timing state for a fresh session.

    Returns:
        None
    """
    st.session_state.messages = []
    st.session_state.docs = []
    st.session_state.queued_message = None
    st.session_state.last_prompt = None
    st.session_state.prev_question_timestamp = datetime.fromtimestamp(0)
    st.session_state.chat_input = ""


def get_history(n_turns=HISTORY_LENGTH):
    """Return the last n user+assistant message pairs for context.

    Args:
        n_turns: Number of conversational turns (user+assistant) to keep.

    Returns:
        list[dict]: Recent messages with roles and content.
    """
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-(n_turns * 2) :]
    ]


def group_sources(docs):
    """Group retrieved pages by title + QID.

    Args:
        docs: List of LangChain Documents returned from retrieval.

    Returns:
        dict: Mapping of title to pages list and optional QID.
    """
    grouped = {}
    for doc in docs:
        meta = doc.metadata
        title = meta.get("title") or meta.get("source") or "Unknown Document"
        page = meta.get("page", "Unknown")
        qid = meta.get("qid")

        if title not in grouped:
            grouped[title] = {"pages": set(), "qid": qid}

        grouped[title]["pages"].add(page)

    for k in grouped:
        grouped[k]["pages"] = sorted(list(grouped[k]["pages"]))
    return grouped


def render_suggestions():
    """Render quick-pick suggestion buttons and queue the selected prompt.

    Returns:
        None
    """
    st.caption("Try a quick prompt:")
    cols = st.columns(len(SUGGESTIONS))
    for (label, prompt), col in zip(SUGGESTIONS.items(), cols):
        if col.button(label, use_container_width=True):
            st.session_state.queued_message = prompt


def render_hero():
    """Render the top hero with icon, title, and prompt bar spacing.

    Returns:
        None
    """
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=103)
    else:
        st.markdown("### ❉")

    title_row = st.container(
        horizontal=True,
        vertical_alignment="bottom",
    )
    with title_row:
        st.title(
            "ASK::MARDI Chat Bot",
            anchor=False,
            width="stretch",
        )
    st.caption("Ask about the indexed content. Answers are grounded in Qdrant context.")


st.set_page_config(page_title="ASK::MARDI Chatbot", layout="centered")
apply_layout_styles()

title_col, action_col = st.columns([6, 1])
with title_col:
    render_hero()
with action_col:
    st.button("Restart", on_click=reset_session, type="secondary")

# --- Render existing chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input(
    "Ask a question...",
    key="chat_input",
)

# Suggestions: inline for first question, expander for follow-ups
if not st.session_state.messages:
    render_suggestions()
    st.caption("⚖️ Legal disclaimer")
else:
    with st.expander("Need inspiration?", expanded=False):
        render_suggestions()
    st.caption("⚖️ Legal disclaimer")

user_message = st.session_state.pop("queued_message", None) or user_input

# Guard against accidental re-processing on reruns
is_new_prompt = user_message and user_message != st.session_state.last_prompt

if is_new_prompt:
    logger.info("New prompt received: %s", user_message[:80])
    st.session_state.last_prompt = user_message

    # Rate-limit rapid-fire questions
    now = datetime.now()
    delta = now - st.session_state.prev_question_timestamp
    if delta < MIN_TIME_BETWEEN_REQUESTS:
        time.sleep((MIN_TIME_BETWEEN_REQUESTS - delta).total_seconds())
    st.session_state.prev_question_timestamp = datetime.now()

    # Capture history before adding the new user turn
    history = get_history()

    # Save and render user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    t_chain_start = time.time()
    chain, llm = build_rag_chain()
    logger.info("build_rag_chain completed in %.2fs", time.time() - t_chain_start)

    # Retrieve docs first (so streaming includes them)
    with st.spinner("Retrieving context… 🔍"):
        t_retrieve = time.time()
        rec = chain.invoke({"question": user_message})
        logger.info("Retrieval+formatting completed in %.2fs", time.time() - t_retrieve)
        docs = rec["docs"]
        context = rec["context"]
        st.session_state.docs = docs

    # Build prompt with context + history
    prompt_str = build_prompt(
        question=user_message,
        context=context,
        history=history,
    )

    # === Streaming answer ===
    streamed_text = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        with st.spinner("Thinking… 🤖"):
            for chunk in llm.stream(prompt_str):
                streamed_text += chunk.content
                answer_placeholder.markdown(streamed_text + "▌")
        answer_placeholder.markdown(streamed_text)

    # Add final answer to chat state
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})

# --- Sources Used ---
if st.session_state.docs:
    st.markdown("---")
    st.subheader("Sources Used 📌")

    grouped = group_sources(st.session_state.docs)

    for title, info in grouped.items():
        pages_str = ", ".join(str(p) for p in info["pages"])
        qid = info.get("qid")

        line = f"**{title}** — pages {pages_str}"
        if qid:
            line += (
                f" · [Open in Portal]"
                f"(https://portal.mardi4nfdi.de/wiki/Item:{qid})"
            )
        st.write(line)
