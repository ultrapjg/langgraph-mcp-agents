import os
import streamlit as st

from input_filter import InputFilter
import backend

backend.ensure_event_loop()
backend.init_state()

st.set_page_config(page_title="Chat", page_icon="🧠", layout="wide")

st.title("💬 MCP Tool Chat")

if not st.session_state.session_initialized:
    st.session_state.event_loop.run_until_complete(
        backend.initialize_session(st.session_state.selected_model)
    )

backend.print_message()

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
        tool_placeholder = st.empty()
        text_placeholder = st.empty()
        resp, final_text, final_tool = st.session_state.event_loop.run_until_complete(
            backend.process_query(
                user_query,
                text_placeholder,
                tool_placeholder,
                st.session_state.timeout_seconds,
            )
        )
    if "error" in resp:
        st.error(resp["error"])
    else:
        st.session_state.history.append({"role": "user", "content": user_query})
        st.session_state.history.append({"role": "assistant", "content": final_text})
        if final_tool.strip():
            st.session_state.history.append({"role": "assistant_tool", "content": final_tool})
        st.rerun()

