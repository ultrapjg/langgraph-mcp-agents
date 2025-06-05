import os
import streamlit as st

from backend_client import backend_initialize, backend_process

from input_filter import InputFilter


def init_state() -> None:
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = False
        st.session_state.history = []
        st.session_state.timeout_seconds = 120
        st.session_state.selected_model = "claude-3-7-sonnet-latest"
        st.session_state.recursion_limit = 100




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

st.set_page_config(page_title="MCP Chat 사용자 페이지", page_icon="💬", layout="wide")

st.title("💬 MCP Chat 사용자 페이지")

if not st.session_state.session_initialized:
    try:
        backend_initialize(st.session_state.selected_model)
    except Exception as e:
        st.error(f"Backend initialization failed: {e}")
        st.stop()
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
        try:
            resp = backend_process(user_query, st.session_state.timeout_seconds)
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()
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

