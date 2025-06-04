import os
import requests
import streamlit as st

from input_filter import InputFilter

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def init_state() -> None:
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = False
        st.session_state.history = []
        st.session_state.timeout_seconds = 120
        st.session_state.selected_model = "claude-3-7-sonnet-latest"
        st.session_state.recursion_limit = 100


def backend_initialize(selected_model: str) -> bool:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/initialize",
            json={"selected_model": selected_model},
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


def backend_process(query: str, timeout_seconds: int) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/query",
        json={"query": query, "timeout_seconds": timeout_seconds},
        timeout=timeout_seconds + 5,
    )
    resp.raise_for_status()
    return resp.json()


def print_message() -> None:
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1


init_state()

st.set_page_config(page_title="Chat", page_icon="🧠", layout="wide")

st.title("💬 MCP Tool Chat")

if not st.session_state.session_initialized:
    backend_initialize(st.session_state.selected_model)
    st.session_state.session_initialized = True

print_message()

user_query = st.chat_input("💬 Enter your question")
if user_query:
    if InputFilter.contains_sensitive(user_query):
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        st.chat_message("assistant", avatar="🤖").warning(
            "❌ Sensitive information detected in input. Processing has been halted."
        )
        st.stop()
    st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
    with st.chat_message("assistant", avatar="🤖"):
        resp = backend_process(user_query, st.session_state.timeout_seconds)
        final_text = resp.get("text", "")
        final_tool = resp.get("tool", "")
        st.markdown(final_text)
        if final_tool.strip():
            with st.expander("🔧 Tool Call Information", expanded=False):
                st.markdown(final_tool)
    st.session_state.history.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "assistant", "content": final_text})
    if final_tool.strip():
        st.session_state.history.append({"role": "assistant_tool", "content": final_tool})
    st.rerun()

