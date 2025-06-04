import json
import os
import requests
import streamlit as st

from input_filter import InputFilter, FILTER_CONFIG_PATH, load_filter_rules
from utils import random_uuid

CONFIG_FILE_PATH = "config.json"
DEFAULT_CONFIG = {
    "get_current_time": {
        "command": "python",
        "args": ["./mcp_server_time.py"],
        "transport": "stdio",
    }
}

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


def init_state() -> None:
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = False
        st.session_state.history = []
        st.session_state.timeout_seconds = 120
        st.session_state.selected_model = "claude-3-7-sonnet-latest"
        st.session_state.recursion_limit = 100


def load_config_from_json() -> dict:
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
            return DEFAULT_CONFIG
    except Exception:
        return DEFAULT_CONFIG


def save_config_to_json(config: dict) -> bool:
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def backend_initialize(selected_model: str, mcp_config: dict) -> bool:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/initialize",
            json={"selected_model": selected_model, "mcp_config": mcp_config},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        st.session_state.tool_count = data.get("tool_count", 0)
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

# initialize session state
init_state()

# login handling
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

if use_login and not st.session_state.authenticated:
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠")
else:
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

if use_login and not st.session_state.authenticated:
    st.title("🔐 Login")
    st.markdown("Login is required to use the system.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if (
                username == os.environ.get("USER_ID")
                and password == os.environ.get("USER_PASSWORD")
            ):
                st.session_state.authenticated = True
                st.success("✅ Login successful! Please wait...")
                st.rerun()
            else:
                st.error("❌ Username or password is incorrect.")
    st.stop()

# header
st.sidebar.markdown("#✍️Made by Architecture Team 3 filter UI version 🚀")
st.sidebar.markdown("### 💻 [Project Page](https://github.com/ultrapjg/langgraph-mcp-agents)")
st.sidebar.divider()

st.title("💬 MCP Tool Utilization Agent")
st.markdown("✨ Ask questions to the ReAct agent that utilizes MCP tools.")

# Sidebar: system settings
with st.sidebar:
    st.subheader("⚙️ System Settings")

    available_models = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available_models.extend([
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ])
    if os.environ.get("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o", "gpt-4o-mini"])
    if not available_models:
        st.warning(
            "⚠️ API keys are not configured. Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file."
        )
        available_models = ["claude-3-7-sonnet-latest"]

    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 Select model to use",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models
        else 0,
        help="Anthropic models require ANTHROPIC_API_KEY and OpenAI models require OPENAI_API_KEY to be set as environment variables.",
    )
    if previous_model != st.session_state.selected_model and st.session_state.session_initialized:
        st.warning("⚠️ Model has been changed. Click 'Apply Settings' button to apply changes.")

    def save_filter_rules(rules):
        with open(FILTER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2)

    with st.expander("🛡️ Filter Settings", expanded=False):
        st.subheader("Filter Rules")
        if "pending_filter_rules" not in st.session_state:
            st.session_state.pending_filter_rules = load_filter_rules()
        with st.form("add_filter_form"):
            new_name = st.text_input("Rule Name", "")
            new_pattern = st.text_input("Regex Pattern", "")
            if st.form_submit_button("➕ Add Rule"):
                if new_name and new_pattern:
                    st.session_state.pending_filter_rules.append({"name": new_name, "pattern": new_pattern})
                    st.success(f"Added rule: {new_name}")
                    st.rerun()
                else:
                    st.error("Both name and pattern are required.")
        st.markdown("**Current rules:**")
        for idx, rule in enumerate(st.session_state.pending_filter_rules):
            col1, col2, col3 = st.columns([2, 6, 1])
            col1.write(rule.get("name", ""))
            col2.code(rule.get("pattern", ""))
            if col3.button("❌", key=f"del_{idx}"):
                removed = st.session_state.pending_filter_rules.pop(idx)
                st.success(f"Removed rule: {removed.get('name')}")
                st.rerun()
        if st.button("✅ Apply Filter Settings"):
            save_filter_rules(st.session_state.pending_filter_rules)
            st.success("filter-config.json has been updated.")

    with st.expander("📋 Registered Filters List", expanded=True):
        st.subheader("Saved Filter Rules")
        try:
            rules = load_filter_rules()
        except Exception:
            st.error("⚠️ Unable to load filter-config.json")
        else:
            if not rules:
                st.info("No filter rules defined.")
            for idx, rule in enumerate(rules):
                col1, col2 = st.columns([4, 8])
                col1.markdown(f"**{rule.get('name', '<no name>')}**")
                col2.code(rule.get('pattern', '<no pattern>'))

    st.session_state.timeout_seconds = st.slider(
        "⏱️ Response generation time limit (seconds)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="Set the maximum time for the agent to generate a response. Complex tasks may require more time.",
    )
    st.session_state.recursion_limit = st.slider(
        "⏱️ Recursion call limit (count)",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
        help="Set the recursion call limit. Setting too high a value may cause memory issues.",
    )

    st.divider()
    st.subheader("🔧 Tool Settings")
    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    with st.expander("Add MCP Tools", expanded=st.session_state.mcp_tools_expander):
        if "pending_mcp_config" not in st.session_state:
            st.session_state.pending_mcp_config = load_config_from_json()
        st.session_state.mcp_config_text = st.text_area(
            "Tool JSON",
            json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False),
            height=300,
        )
        if st.button("Load", key="load_config"):
            try:
                loaded = json.loads(st.session_state.mcp_config_text)
            except Exception:
                st.error("Invalid JSON")
            else:
                st.session_state.pending_mcp_config = loaded
                st.success("Configuration loaded.")
                st.rerun()
        if st.button("Clear", key="clear_config"):
            st.session_state.pending_mcp_config = {}
            st.session_state.mcp_config_text = "{}"
            st.rerun()
        if st.button("Add from File", key="file_load"):
            uploaded = st.file_uploader("Upload JSON file", type=["json"])
            if uploaded:
                try:
                    st.session_state.pending_mcp_config.update(json.load(uploaded))
                    st.session_state.mcp_config_text = json.dumps(
                        st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
                    )
                    st.success("Loaded from file.")
                except Exception:
                    st.error("Failed to load file.")
        for tool_name in list(st.session_state.pending_mcp_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("Delete", key=f"delete_{tool_name}"):
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} tool has been deleted. Click 'Apply Settings' button to apply.")

    st.divider()

# Sidebar: info and action buttons
with st.sidebar:
    st.subheader("📊 System Information")
    st.write(f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}")
    st.write(f"🧠 Current Model: {st.session_state.selected_model}")
    if st.button("Apply Settings", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 Applying changes. Please wait...")
            progress = st.progress(0)
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )
            saved = save_config_to_json(st.session_state.pending_mcp_config)
            if not saved:
                st.error("❌ Failed to save settings file.")
            progress.progress(15)
            st.session_state.session_initialized = False
            progress.progress(30)
            success = backend_initialize(
                st.session_state.selected_model, st.session_state.pending_mcp_config
            )
            progress.progress(100)
            if success:
                st.session_state.session_initialized = True
                st.success("✅ New settings have been applied.")
                st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ Failed to apply settings.")
        st.rerun()
    st.divider()
    st.subheader("🔄 Actions")
    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = random_uuid()
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()
    if use_login and st.session_state.authenticated:
        st.divider()
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.success("✅ You have been logged out.")
            st.rerun()

if not st.session_state.session_initialized:
    st.info("MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize.")

print_message()

user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
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
    else:
        st.warning("⚠️ MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize.")

