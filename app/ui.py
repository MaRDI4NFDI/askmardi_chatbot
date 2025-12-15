import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from app.rag_chain import build_rag_chain
from app.prompts import build_prompt
from app.logger import get_logger
from app.config import check_config, load_config

HISTORY_LENGTH = 5
MIN_TIME_BETWEEN_REQUESTS = timedelta(seconds=3)
EMPTY_ANSWER_FALLBACK = (
    "Sorry, I couldn't generate a response this time. Please try again or "
    "rephrase your question."
)
SUGGESTIONS = {
    "🔔 Regularization in glmnet": (
        "How does the glmnet package implement regularization for linear and logistic regression?"
    ),
    "🌦️ Network analysis": (
        "Show available packages for graph and network analysis and how measures such as centrality " +
        "or connectivity are defined."
    ),
    "🧮 Time-series models": (
        "How are time-series models mathematically represented, and "+
        "which criteria are used to compare or select competing models according to the package documentation?"
    ),
}

DISCLAIMER_TEXT = """This AI chatbot is powered by the MaRDI Knowledge Graph.
Keep in mind that this is a very early prototype and only a fraction of the 
knowledge graph has actually been indexed yet.
Answers may be inaccurate, inefficient, or biased. Any use or decisions
based on such answers should include reasonable practices including human
oversight to ensure they are safe, accurate, and suitable for your intended
purpose. The MaRDI consortium is not liable for any actions, losses, or damages
resulting from the use of the chatbot. Do not enter any private, sensitive,
personal, or regulated data. By using this chatbot, you acknowledge and agree
that input you provide and answers you receive (collectively, "Content") may
be used by MaRDI to provide, maintain, develop, and improve their
respective offerings. For more information on how MaRDI may use your
content, see https://www.mardi4nfdi.de .
"""


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
config_ok = check_config()


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
    if input_tokens >= threshold:
        logger.warning(
            "LLM input tokens (%d) exceeded threshold (%d)",
            input_tokens,
            threshold,
        )
    else:
        logger.debug(
            "LLM input tokens (%d) within threshold (%d)",
            input_tokens,
            threshold,
        )
    return input_tokens, output_tokens, total_tokens


def estimate_token_count(text: str) -> int:
    """Return a quick token estimate using a mixed heuristic.

    Args:
        text: Input string to approximate token count for.

    Returns:
        int: Approximate token count.
    """
    # Heuristic: count word/punctuation chunks and also scale by 4 chars per token,
    # then take the max to avoid undercounting.
    rough_bpe = max(1, len(text) // 4)
    wordish = len(re.findall(r"\w+|[^\w\s]", text))
    return max(rough_bpe, wordish)


def _ns_to_seconds(ns_value):
    """Convert nanoseconds to seconds if present.

    Args:
        ns_value: Duration in nanoseconds or None.

    Returns:
        float | None: Duration in seconds if convertible, else None.
    """
    return ns_value / 1_000_000_000 if ns_value is not None else None


def _format_duration(seconds):
    """Format a duration in seconds for logging.

    Args:
        seconds: Duration in seconds or None.

    Returns:
        str: Formatted string like ``0.12s`` or ``n/a`` if missing.
    """
    return f"{seconds:.2f}s" if seconds is not None else "n/a"


def apply_layout_styles():
    """Constrain page width and add comfortable padding.

    Returns:
        None
    """
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1100px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stChatInput"] {
            border: 2px solid #f5a623;
            border-radius: 14px;
            padding: 0.35rem 0.85rem;
            box-shadow: 0 6px 20px rgba(245, 166, 35, 0.25);
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
        st.image(str(logo_path), width=150)
    else:
        st.markdown("### ❉")

    title_row = st.container(
        horizontal=True,
        vertical_alignment="bottom",
    )
    with title_row:
        st.title(
            "Your AI for the MaRDI KG",
            anchor=False,
            width="stretch",
        )
    st.caption("You can ask about the currently indexed content from the MaRDI knowledge graph.")


@st.dialog("Legal Disclaimer")
def show_disclaimer_dialog():
    """Render the modal containing the legal disclaimer text.

    Returns:
        None
    """
    st.markdown(DISCLAIMER_TEXT)


def render_disclaimer():
    """Render a clickable control that opens the legal disclaimer modal.

    Returns:
        None
    """
    if st.button("Disclaimer", type="secondary", use_container_width=True):
        show_disclaimer_dialog()


st.set_page_config(page_title="ASK::MARDI Chatbot", layout="centered")
apply_layout_styles()

title_col, action_col = st.columns([9, 2])
with title_col:
    render_hero()
with action_col:
    st.button("Restart", on_click=reset_session, type="secondary", use_container_width=True)
    render_disclaimer()

if not config_ok:
    st.error("Config checks failed. Qdrant or LLM not reachable.")
    st.stop()

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
else:
    with st.expander("Need inspiration?", expanded=False):
        render_suggestions()

user_message = st.session_state.pop("queued_message", None) or user_input

# Guard against accidental re-processing on reruns
is_new_prompt = user_message and user_message != st.session_state.last_prompt

if is_new_prompt:
    request_start = time.time()
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
    chain_build_duration = time.time() - t_chain_start
    logger.info("build_rag_chain completed in %.2fs", chain_build_duration)

    # Retrieve docs first (so streaming includes them)
    with st.spinner("Retrieving context… 🔍"):
        t_retrieve = time.time()

        try:
            rec = chain.invoke({"question": user_message})
        except Exception as e:
            logger.exception("Could not connect to qdrant server: %s", e)
            st.error("Could not connect to qdrant server:: %s" % e)
            st.stop()

        retrieval_duration = time.time() - t_retrieve
        logger.info("Retrieval+formatting completed in %.2fs", retrieval_duration)
        docs = rec["docs"]
        context = rec["context"]
        st.session_state.docs = docs

    # Build prompt with context + history
    prompt_str = build_prompt(
        question=user_message,
        context=context,
        history=history,
    )

    logger.info("LLM Query: %s", prompt_str[-1500:])
    logger.info(
        "LLM Query total characters: %s (approx tokens=%s)",
        len(prompt_str),
        estimate_token_count(prompt_str),
    )

    # === Streaming answer ===
    streamed_text = ""
    chunk_count = 0
    empty_chunk_count = 0
    first_chunks = []
    last_chunk = None
    meta_chunk = None
    last_chunk_dump = None
    last_usage = None
    finish_reason = None
    t_stream = time.time()
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        with st.spinner("Thinking ..."):
            try:
                for chunk in llm.stream(prompt_str):
                    last_chunk = chunk
                    if getattr(chunk, "response_metadata", None):
                        meta_chunk = chunk
                    try:
                        last_chunk_dump = chunk.model_dump(exclude_none=True)
                    except Exception:
                        last_chunk_dump = None
                    if getattr(chunk, "usage_metadata", None):
                        last_usage = chunk.usage_metadata
                    chunk_count += 1
                    content = chunk.content or ""
                    if not finish_reason:
                        finish_reason = getattr(
                            chunk, "response_metadata", {}
                        ).get("finish_reason")
                    if chunk_count == 1:
                        logger.debug(
                            "First chunk meta=%s extra=%s raw=%r",
                            getattr(chunk, "response_metadata", None),
                            getattr(chunk, "additional_kwargs", None),
                            chunk,
                        )
                        logger.info(
                            "Received first LLM chunk (%s chars)", len(content)
                        )
                    if not content:
                        empty_chunk_count += 1
                        if len(first_chunks) < 5:
                            first_chunks.append(
                                {
                                    "n": chunk_count,
                                    "len": len(content),
                                    "meta": getattr(
                                        chunk, "response_metadata", None
                                    ),
                                    "extra": getattr(
                                        chunk, "additional_kwargs", None
                                    ),
                                    "raw": repr(chunk),
                                    "dump": last_chunk_dump,
                                }
                            )
                        if empty_chunk_count <= 3:
                            logger.debug(
                                "Empty chunk #%d meta=%s extra=%s raw=%r",
                                empty_chunk_count,
                                getattr(chunk, "response_metadata", None),
                                getattr(chunk, "additional_kwargs", None),
                                chunk,
                            )
                        if (
                            getattr(chunk, "usage_metadata", None)
                            and chunk.usage_metadata.get("output_tokens")
                        ):
                            usage_line = (
                                f"Empty chunk carried usage: {chunk.usage_metadata}"
                            )
                            logger.info(usage_line)
                            inspect_usage_line(usage_line)
                    streamed_text += content
                    answer_placeholder.markdown(streamed_text + "▌")
            except Exception:
                logger.exception(
                    "LLM streaming failed after %d chunks (%.2fs)",
                    chunk_count,
                    time.time() - t_stream,
                )
                raise
        answer_placeholder.markdown(streamed_text)
    stream_duration = time.time() - t_stream
    if not streamed_text:
        logger.warning(
            "LLM stream yielded no content (chunks=%d, %.2fs)",
            chunk_count,
            stream_duration,
        )
        if last_usage:
            usage_line = f"LLM usage (last seen): {last_usage}"
            logger.warning(usage_line)
            inspect_usage_line(usage_line)
        if finish_reason:
            logger.warning("LLM finish_reason: %s", finish_reason)
        if first_chunks:
            logger.debug(
                "First chunks detail (up to 5): %s",
                first_chunks,
            )
        if last_chunk:
            logger.debug(
                "Last chunk meta=%s extra=%s raw=%r dump=%s",
                getattr(last_chunk, "response_metadata", None),
                getattr(last_chunk, "additional_kwargs", None),
                last_chunk,
                last_chunk_dump,
            )
        streamed_text = EMPTY_ANSWER_FALLBACK
        answer_placeholder.markdown(streamed_text)
    else:
        logger.info(
            "LLM stream completed in %.2fs (chunks=%d, chars=%d)",
            stream_duration,
            chunk_count,
            len(streamed_text),
        )
        if empty_chunk_count:
            logger.info(
                "LLM stream had %d empty chunks out of %d",
                empty_chunk_count,
                chunk_count,
            )
        if last_usage:
            usage_line = f"LLM usage (last seen): {last_usage}"
            logger.info(usage_line)
            inspect_usage_line(usage_line)
        if finish_reason:
            logger.debug("LLM finish_reason: %s", finish_reason)
        if first_chunks:
            logger.debug(
                "First chunks detail (up to 5): %s",
                first_chunks,
            )
        if last_chunk:
            logger.debug(
                "Last chunk meta=%s extra=%s raw=%r dump=%s",
                getattr(last_chunk, "response_metadata", None),
                getattr(last_chunk, "additional_kwargs", None),
                last_chunk,
                last_chunk_dump,
            )

    # Add final answer to chat state
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})
    total_duration = time.time() - request_start
    source_meta_chunk = meta_chunk or last_chunk
    meta = getattr(source_meta_chunk, "response_metadata", {}) if source_meta_chunk else {}
    meta_total = _ns_to_seconds(meta.get("total_duration"))
    meta_load = _ns_to_seconds(meta.get("load_duration"))
    meta_prompt_eval = _ns_to_seconds(meta.get("prompt_eval_duration"))
    meta_eval = _ns_to_seconds(meta.get("eval_duration"))
    logger.info(
        (
            "LLM Answer: %s "
            "(model_total=%s, load=%s, prompt_eval=%s, eval=%s, "
            "retrieve=%.2fs, chain_build=%.2fs, total=%.2fs)"
        ),
        streamed_text[:80],
        _format_duration(meta_total),
        _format_duration(meta_load),
        _format_duration(meta_prompt_eval),
        _format_duration(meta_eval),
        retrieval_duration,
        chain_build_duration,
        total_duration,
    )

# --- Sources Used ---
if st.session_state.docs:
    st.markdown("---")
    st.subheader("Sources Used")

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
