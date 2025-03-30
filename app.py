import streamlit as st
import asyncio
import nest_asyncio
import json

# Apply nest_asyncio: Allow nested calls within an already running event loop
nest_asyncio.apply()

# Create and reuse global event loop (create once and continue using)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_teddynote.messages import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Load environment variables (get API keys and settings from .env file)
load_dotenv(override=True)

# Page configuration: title, icon, layout
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# Add author information at the top of the sidebar (placed before other sidebar elements)
st.sidebar.markdown("### ✍️ Made by [TeddyNote](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()  # Add divider

# Page title and description
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ Ask questions to the ReAct agent using MCP tools.")

# Initialize session state
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # Session initialization flag
    st.session_state.agent = None  # Storage for ReAct agent object
    st.session_state.history = []  # List for storing conversation history
    st.session_state.mcp_client = None  # Storage for MCP client object

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- Function Definitions ---


def print_message():
    """
    Display chat history on the screen.

    Distinguishes between user and assistant messages,
    and displays tool call information in expandable panels.
    """
    for message in st.session_state.history:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").markdown(message["content"])
        elif message["role"] == "assistant_tool":
            with st.expander("🔧 Tool Call Information", expanded=False):
                st.markdown(message["content"])


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Create a streaming callback function.

    Parameters:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: Streaming callback function
        accumulated_text: List to store accumulated text responses
        accumulated_tool: List to store accumulated tool call information
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander(
                        "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    Process user questions and generate responses.

    Parameters:
        query: Text of the question entered by the user
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation time limit (seconds)

    Returns:
        response: Agent's response object
        final_text: Final text response
        final_tool: Final tool call information
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=100, thread_id=st.session_state.thread_id
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 Agent has not been initialized."},
                "🚫 Agent has not been initialized.",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ Error processing query: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    Initialize MCP session and agent.

    Parameters:
        mcp_config: MCP tool configuration (JSON). Use default settings if None

    Returns:
        bool: Initialization success status
    """
    try:
        with st.spinner("🔄 Connecting to MCP server..."):
            if mcp_config is None:
                # Use default settings
                mcp_config = {
                    "weather": {
                        "command": "python",
                        "args": ["./mcp_server_local.py"],
                        "transport": "stdio",
                    },
                }
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            model = ChatAnthropic(
                model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000
            )
            agent = create_react_agent(
                model,
                tools,
                checkpointer=MemorySaver(),
                prompt="Use your tools to answer the question.",
            )
            st.session_state.agent = agent
            st.session_state.session_initialized = True
            return True
    except Exception as e:
        st.error(f"❌ Error during initialization: {str(e)}")
        import traceback

        st.error(traceback.format_exc())
        return False


# --- Sidebar UI: Changed to MCP tool addition interface ---
with st.sidebar.expander("Add MCP Tool", expanded=False):
    default_config = """{
  "weather": {
    "command": "python",
    "args": ["./mcp_server_local.py"],
    "transport": "stdio"
  }
}"""
    # Create pending config based on existing mcp_config_text if not present
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(
                st.session_state.get("mcp_config_text", default_config)
            )
        except Exception as e:
            st.error(f"Failed to set initial pending config: {e}")

    # UI for adding individual tools
    st.subheader("Add Individual Tool")
    st.markdown(
        """
    Enter **one tool** in JSON format:
    
    ```json
    {
      "tool_name": {
        "command": "execution_command",
        "args": ["arg1", "arg2", ...],
        "transport": "stdio"
      }
    }
    ```    
    ⚠️ **Important**: JSON must be wrapped in curly braces (`{}`).
    """
    )

    # Provide clearer examples
    example_json = {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@smithery-ai/github",
                "--config",
                '{"githubPersonalAccessToken":"your_token_here"}',
            ],
            "transport": "stdio",
        }
    }

    default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

    new_tool_json = st.text_area(
        "Tool JSON",
        default_text,
        height=250,
    )

    # Add button
    if st.button("Add Tool"):
        try:
            # Validate input
            if not new_tool_json.strip().startswith(
                "{"
            ) or not new_tool_json.strip().endswith("}"):
                st.error("JSON must start and end with curly braces ({}).")
                st.markdown('Correct format: `{ "tool_name": { ... } }`')
            else:
                # Parse JSON
                parsed_tool = json.loads(new_tool_json)

                # Check if it's in mcpServers format and process
                if "mcpServers" in parsed_tool:
                    # Move contents of mcpServers to top level
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' format detected. Converting automatically.")

                # Check number of tools entered
                if len(parsed_tool) == 0:
                    st.error("Please enter at least one tool.")
                else:
                    # Process all tools
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # Check URL field and set transport
                        if "url" in tool_config:
                            # Set transport to "sse" if URL exists
                            tool_config["transport"] = "sse"
                            st.info(
                                f"URL detected in '{tool_name}' tool, setting transport to 'sse'."
                            )
                        elif "transport" not in tool_config:
                            # Set default "stdio" if URL doesn't exist and transport is not set
                            tool_config["transport"] = "stdio"

                        # Check required fields
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(
                                f"'{tool_name}' tool configuration requires 'command' or 'url' field."
                            )
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(
                                f"'{tool_name}' tool configuration requires 'args' field."
                            )
                        elif "command" in tool_config and not isinstance(
                            tool_config["args"], list
                        ):
                            st.error(
                                f"'args' field in '{tool_name}' tool must be in array ([]) format."
                            )
                        else:
                            # Add tool to pending_mcp_config
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    # Success message
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(
                                f"{success_tools[0]} tool has been added. Press 'Apply' button to apply."
                            )
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(
                                f"Total {len(success_tools)} tools ({tool_names}) have been added. Press 'Apply' button to apply."
                            )
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            st.markdown(
                f"""
            **How to fix**:
            1. Check if the JSON format is correct.
            2. All keys must be wrapped in double quotes (").
            3. String values must also be wrapped in double quotes (").
            4. Double quotes within strings must be escaped (\\").
            """
            )
        except Exception as e:
            st.error(f"Error occurred: {e}")

    # Add divider
    st.divider()

    # Display current tool settings (read-only)
    st.subheader("Current Tool Settings (Read-only)")
    st.code(
        json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
    )

# --- Display registered tools list and add delete buttons ---
with st.sidebar.expander("Registered Tools List", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("Not a valid MCP tool configuration.")
    else:
        # Iterate through keys (tool names) in pending config
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("Delete", key=f"delete_{tool_name}"):
                # Delete the tool from pending config (not applied immediately)
                del st.session_state.pending_mcp_config[tool_name]
                st.success(
                    f"{tool_name} tool has been deleted. Press 'Apply' button to apply."
                )

with st.sidebar:

    # Apply button: Apply pending config to actual settings and reinitialize session
    if st.button(
        "Apply Tool Configurations",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # Display applying message
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 Applying changes. Please wait...")
            progress_bar = st.progress(0)

            # Save settings
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # Prepare session initialization
            st.session_state.session_initialized = False
            st.session_state.agent = None
            st.session_state.mcp_client = None

            # Update progress status
            progress_bar.progress(30)

            # Run initialization
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # Update progress status
            progress_bar.progress(100)

            if success:
                st.success("✅ New MCP tool settings have been applied.")
            else:
                st.error("❌ Failed to apply new MCP tool settings.")

        # Refresh page
        st.rerun()


# --- Basic session initialization (if not initialized) ---
if not st.session_state.session_initialized:
    st.info("🔄 Initializing MCP server and agent. Please wait...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(
            f"✅ Initialization complete! {st.session_state.tool_count} tools loaded."
        )
    else:
        st.error("❌ Initialization failed. Please refresh the page.")


# --- Display conversation history ---
print_message()

# --- User input and processing ---
user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(user_query, text_placeholder, tool_placeholder)
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning("⏳ System is still initializing. Please try again in a moment.")

# --- Sidebar: Display system information ---
with st.sidebar:
    st.subheader("🔧 System Information")
    st.write(
        f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}"
    )
    st.write("🧠 Model: Claude 3.7 Sonnet")

    # Add divider (visual separation)
    st.divider()

    # Add conversation reset button at the bottom of sidebar
    if st.button("🔄 Reset Conversation", use_container_width=True, type="primary"):
        # Reset thread_id
        st.session_state.thread_id = random_uuid()

        # Reset conversation history
        st.session_state.history = []

        # Notification message
        st.success("✅ Conversation has been reset.")

        # Refresh page
        st.rerun()
