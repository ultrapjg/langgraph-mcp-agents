import streamlit as st
import asyncio
import nest_asyncio
import json
import os

# Apply nest_asyncio to allow nested event loop calls within an already running event loop
nest_asyncio.apply()

# Create and reuse a global event loop (create once, use throughout the session)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Load environment variables (API keys and other settings from .env file)
load_dotenv(override=True)

# Page configuration: title, icon, and layout
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# Add author information at the top of the sidebar (placed before other sidebar elements)
st.sidebar.markdown("### ✍️ Made by [TeddyNote](https://youtube.com/c/teddynote) 🚀")
st.sidebar.markdown(
    "### 💻 [Project Page](https://github.com/teddynote-lab/langgraph-mcp-agents)"
)
st.sidebar.divider()  # Add a divider

# Page title and description
st.title("💬 Agent with MCP Tools")
st.markdown("✨ Ask questions to the ReAct agent that utilizes MCP tools.")

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
}

# Initialize session state
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = (
        False  # Flag for session initialization status
    )
    st.session_state.agent = None  # Storage for ReAct agent object
    st.session_state.history = []  # List for conversation history
    st.session_state.mcp_client = None  # Storage for MCP client object
    st.session_state.timeout_seconds = (
        120  # Response generation timeout in seconds, default 120s
    )
    st.session_state.selected_model = (
        "claude-3-7-sonnet-latest"  # Default model selection
    )
    st.session_state.recursion_limit = 100  # Recursion call limit, default 100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- Function Definitions ---


async def cleanup_mcp_client():
    """
    Safely terminates the existing MCP client.

    This function properly releases resources if an existing client is present.
    It ensures that connections are closed properly to prevent resource leaks.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

            # st.warning(f"Error while terminating MCP client: {str(e)}")
            # st.warning(traceback.format_exc())


def print_message():
    """
    Displays the chat history on the screen.

    This function renders user and assistant messages with appropriate styling.
    Tool call information is displayed within the assistant message container.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # Create assistant message container
            with st.chat_message("assistant", avatar="🤖"):
                # Display assistant message content
                st.markdown(message["content"])

                # Check if the next message is tool call information
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # Display tool call information in the same container as an expander
                    with st.expander("🔧 Tool Call Information", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # Increment by 2 as we've processed two messages together
                else:
                    i += 1  # Increment by 1 as we've only processed one message
        else:
            # Skip assistant_tool messages as they are handled above
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Creates a streaming callback function for real-time response display.

    This function generates a callback that displays LLM-generated responses in real-time
    on the screen. It handles both text responses and tool call information separately.

    Args:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: The streaming callback function
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
            # Handle content in list format (common with Claude models)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # Handle text type content
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # Handle tool use type content
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
            # Handle tool_calls attribute (common with OpenAI models)
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Handle simple string content
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # Handle invalid tool calls
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Invalid Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Handle tool_call_chunks attribute
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Handle tool_calls in additional_kwargs (for compatibility with various models)
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        # Handle tool messages (tool responses)
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
    Processes user queries and generates responses.

    This function sends the user's question to the agent and streams the response in real-time.
    It returns a timeout error if the response is not completed within the specified time.

    Args:
        query: The text of the user's question
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation timeout in seconds

    Returns:
        response: The agent's response object
        final_text: The final text response
        final_tool: The final tool call information
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
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ Request exceeded the {timeout_seconds} second time limit. Please try again later."
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
    Initializes the MCP session and agent.

    This function sets up the MCP client and creates a ReAct agent with the specified tools.
    It handles the connection to MCP servers and configures the language model.

    Args:
        mcp_config: MCP tool configuration (JSON). Uses default settings if None

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    with st.spinner("🔄 Connecting to MCP servers..."):
        # First, safely clean up any existing client
        await cleanup_mcp_client()

        if mcp_config is None:
            # Use default configuration
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

        # Initialize appropriate model based on selection
        selected_model = st.session_state.selected_model
        # Define a dictionary to store token limits for different models
        # This helps manage the maximum output tokens for each supported model
        OUTPUT_TOKEN_INFO = {
            # Anthropic models
            "claude-3-7-sonnet-latest": {"max_tokens": 4096},
            "claude-3-5-sonnet-latest": {"max_tokens": 4096},
            "claude-3-5-haiku-latest": {"max_tokens": 4096},
            # OpenAI models
            "gpt-4o": {"max_tokens": 4096},
            "gpt-4o-mini": {"max_tokens": 4096},
        }

        # Initialize the appropriate language model based on user selection
        if selected_model in [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        else:  # Use OpenAI model
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True


# --- Sidebar: System Settings Section ---
with st.sidebar:
    st.subheader("⚙️ System Settings")

    # Model selection feature
    # Create a list of available models
    available_models = []

    # Check for Anthropic API key
    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    if has_anthropic_key:
        available_models.extend(
            [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]
        )

    # Check for OpenAI API key
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    if has_openai_key:
        available_models.extend(["gpt-4o", "gpt-4o-mini"])

    # Display a message if no models are available
    if not available_models:
        st.warning(
            "⚠️ No API keys configured. Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file."
        )
        # Add Claude model as default (to show UI even without keys)
        available_models = ["claude-3-7-sonnet-latest"]

    # Model selection dropdown
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 Select Model",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        ),
        help="Anthropic models require ANTHROPIC_API_KEY and OpenAI models require OPENAI_API_KEY as environment variables.",
    )

    # Notify when model is changed and session needs reinitialization
    if (
        previous_model != st.session_state.selected_model
        and st.session_state.session_initialized
    ):
        st.warning(
            "⚠️ Model has been changed. Click 'Apply Settings' button to apply changes."
        )

    # Timeout setting slider
    st.session_state.timeout_seconds = st.slider(
        "⏱️ Response Generation Timeout (seconds)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="Set the maximum time for the agent to generate a response. Complex tasks may require more time.",
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ Recursion Call Limit (count)",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
        help="Set the recursion call limit. Setting too high a value may cause memory issues.",
    )

    st.divider()  # Add divider

    # Tools configuration section
    st.subheader("🔧 Tool Settings")

    # Manage expander state in session state
    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    # MCP tool addition interface
    with st.expander("🧰 Add MCP Tools", expanded=st.session_state.mcp_tools_expander):
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
                st.error(f"Failed to initialize pending config: {e}")

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

        # Provide a clearer example
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
        if st.button(
            "Add Tool",
            type="primary",
            key="add_tool_button",
            use_container_width=True,
        ):
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

                    # Check if it's in mcpServers format and process accordingly
                    if "mcpServers" in parsed_tool:
                        # Move contents of mcpServers to top level
                        parsed_tool = parsed_tool["mcpServers"]
                        st.info(
                            "'mcpServers' format detected. Converting automatically."
                        )

                    # Check number of tools entered
                    if len(parsed_tool) == 0:
                        st.error("Please enter at least one tool.")
                    else:
                        # Process all tools
                        success_tools = []
                        for tool_name, tool_config in parsed_tool.items():
                            # Check URL field and set transport
                            if "url" in tool_config:
                                # Set transport to "sse" if URL is present
                                tool_config["transport"] = "sse"
                                st.info(
                                    f"URL detected in '{tool_name}' tool, setting transport to 'sse'."
                                )
                            elif "transport" not in tool_config:
                                # Set default "stdio" if no URL and no transport
                                tool_config["transport"] = "stdio"

                            # Check required fields
                            if (
                                "command" not in tool_config
                                and "url" not in tool_config
                            ):
                                st.error(
                                    f"'{tool_name}' tool configuration requires either 'command' or 'url' field."
                                )
                            elif "command" in tool_config and "args" not in tool_config:
                                st.error(
                                    f"'{tool_name}' tool configuration requires 'args' field."
                                )
                            elif "command" in tool_config and not isinstance(
                                tool_config["args"], list
                            ):
                                st.error(
                                    f"'args' field in '{tool_name}' tool must be an array ([])."
                                )
                            else:
                                # Add tool to pending_mcp_config
                                st.session_state.pending_mcp_config[tool_name] = (
                                    tool_config
                                )
                                success_tools.append(tool_name)

                        # Success message
                        if success_tools:
                            if len(success_tools) == 1:
                                st.success(
                                    f"{success_tools[0]} tool has been added. Click 'Apply Settings' button to apply changes."
                                )
                            else:
                                tool_names = ", ".join(success_tools)
                                st.success(
                                    f"Total {len(success_tools)} tools ({tool_names}) have been added. Click 'Apply Settings' button to apply changes."
                                )
                            # Collapse expander after adding
                            st.session_state.mcp_tools_expander = False
                            st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {e}")
                st.markdown(
                    f"""
                **How to fix**:
                1. Check that your JSON format is correct.
                2. All keys must be wrapped in double quotes (").
                3. String values must also be wrapped in double quotes (").
                4. Double quotes within strings must be escaped (\\").
                """
                )
            except Exception as e:
                st.error(f"Error occurred: {e}")

    # Display registered tools list and add delete buttons
    with st.expander("📋 Registered Tools List", expanded=True):
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
                    # Delete tool from pending config (not applied immediately)
                    del st.session_state.pending_mcp_config[tool_name]
                    st.success(
                        f"{tool_name} tool has been deleted. Click 'Apply Settings' button to apply changes."
                    )

    st.divider()  # Add divider

# --- Sidebar: System Information and Action Buttons Section ---
with st.sidebar:
    st.subheader("📊 System Information")
    st.write(
        f"🛠️ MCP Tools Count: {st.session_state.get('tool_count', 'Initializing...')}"
    )
    selected_model_name = st.session_state.selected_model
    st.write(f"🧠 Current Model: {selected_model_name}")

    # Move Apply Settings button here
    if st.button(
        "Apply Settings",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # Show applying message
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

            # Update progress
            progress_bar.progress(30)

            # Run initialization
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # Update progress
            progress_bar.progress(100)

            if success:
                st.success("✅ New settings have been applied.")
                # Collapse tool addition expander
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ Failed to apply settings.")

        # Refresh page
        st.rerun()

    st.divider()  # Add divider

    # Action buttons section
    st.subheader("🔄 Actions")

    # Reset conversation button
    if st.button("Reset Conversation", use_container_width=True, type="primary"):
        # Reset thread_id
        st.session_state.thread_id = random_uuid()

        # Reset conversation history
        st.session_state.history = []

        # Notification message
        st.success("✅ Conversation has been reset.")

        # Refresh page
        st.rerun()

# --- Default session initialization (if not initialized) ---
if not st.session_state.session_initialized:
    st.info(
        "MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize."
    )


# --- Print conversation history ---
print_message()

# --- User input and processing ---
user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
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
        st.warning(
            "⚠️ MCP server and agent are not initialized. Please click the 'Apply Settings' button in the left sidebar to initialize."
        )
