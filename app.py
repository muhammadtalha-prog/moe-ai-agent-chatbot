import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# 1. Inject Streamlit Secrets into environment variables for Cloud Compatibility
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Load dotenv fallback
load_dotenv()

from agent.config import get_groq_key, get_api_key, GROQ_CHAT_MODEL_NAME
from agent.experts.orchestrator import Orchestrator
from agent.file_handler import read_file, write_file
from agent.security import get_workspace_dir

# Set premium Streamlit page configurations
st.set_page_config(
    page_title="AI Agent Chatbot (MoE)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via markdown
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .st-emotion-cache-1c7n2ka {
        background-color: #1E293B !important;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Cache Orchestrator resource
@st.cache_resource
def load_orchestrator():
    return Orchestrator(vector_store_path="vector_store.json")

orchestrator = load_orchestrator()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    # Separate history structure matching orchestrator expectations
    st.session_state.chat_history = []

# --- SIDEBAR PANELS ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/bot.png", width=80)
    st.title("AI Agent Settings")
    st.caption("Mixture of Experts | MCP | Semantic Memory")
    st.markdown("---")
    
    # API Key status panel
    st.subheader("🔑 API Configurations")
    groq_active = bool(get_groq_key())
    gemini_active = bool(get_api_key())
    
    if groq_active:
        st.success("🟢 Groq API: Connected")
        st.info(f"Model: {GROQ_CHAT_MODEL_NAME}")
    else:
        st.error("🔴 Groq API: Disconnected")
        st.warning("Please configure GROQ_API_KEY in secrets or .env file.")
        if st.secrets:
            st.info(f"Loaded Secrets: {list(st.secrets.keys())}")
        else:
            st.info("No Streamlit Secrets loaded.")
        
    if gemini_active:
        st.success("🟢 Gemini API: Connected (Embeddings)")
    else:
        st.warning("🟡 Gemini API: Disconnected (Fallback Mock Embeddings Active)")
        
    st.markdown("---")
    
    # Vector Database status panel
    st.subheader("🧠 Memory Status")
    docs_count = len(orchestrator.memory.documents)
    st.metric("Vector Store Records", f"{docs_count} items")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Vector DB", use_container_width=True):
            orchestrator.memory.clear()
            st.success("Vector DB wiped!")
            st.rerun()
    with col2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            st.rerun()
            
    st.markdown("---")
    st.markdown("Created by Antigravity AI Agent Chatbot.")

# --- MAIN LAYOUT TABBING ---
tab1, tab2 = st.tabs(["💬 Chatbot Room", "📁 File Actions Panel"])

# Tab 1: Chat interface
with tab1:
    st.markdown("### 🤖 MoE Chatbot Assistant")
    st.caption("Type messages normally. The Orchestrator routes requests using standard queries, system prompts, or memory searches.")

    # Show past chat dialogue
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box
    if user_query := st.chat_input("Ask the agent a question or give an instruction..."):
        # Append User Input
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display User Input
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Call Agent Orchestrator with thinking spinner
        with st.chat_message("assistant"):
            with st.spinner("Agent routing query through experts..."):
                try:
                    response = orchestrator.route_and_process(user_query, st.session_state.chat_history)
                    st.markdown(response)
                    
                    # Store response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing chat: {e}")

# Tab 2: File upload processing
with tab2:
    st.markdown("### 📁 File Processing Expert")
    st.caption("Upload files in any format (e.g. Word, Excel, PDF, CSV, JSON, HTML, YAML, etc.). Extract summaries or rewrite content safely in the same file format.")

    uploaded_file = st.file_uploader(
        "Choose a file to analyze"
    )
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        
        # Safe storage in the workspace folder
        workspace = get_workspace_dir()
        temp_dir = workspace / "uploaded_files"
        temp_dir.mkdir(exist_ok=True)
        
        target_path = temp_dir / filename
        
        # Write bytes safely to disk
        with open(target_path, "wb") as f:
            f.write(file_bytes)
            
        st.success(f"Uploaded and secured file: `{filename}`")
        
        # Set panel choices
        action = st.radio(
            "Select action to perform:",
            ["Read & Preview", "Summarize Content", "Rewrite/Modify Content"]
        )
        
        instruction = ""
        output_filename = ""
        if action == "Rewrite/Modify Content":
            instruction = st.text_input(
                "Rewrite instructions:", 
                placeholder="Example: 'translate this file content to French' or 'sanitize raw values'"
            )
            base, ext = os.path.splitext(filename)
            output_filename = st.text_input("Saved Output Filename:", value=f"{base}_modified{ext}")
            
        elif action == "Summarize Content":
            instruction = st.text_input(
                "Summarization focus (optional):", 
                placeholder="Example: 'summarize in 3 core bullet points focused on math'"
            )

        if st.button("🔥 Execute File Action", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    # Translate selection to orchestrator action parameters
                    action_param = ""
                    out_path = None
                    
                    if action == "Read & Preview":
                        action_param = "read"
                    elif action == "Summarize Content":
                        action_param = "summarize"
                    else:
                        action_param = "rewrite"
                        out_path = str(temp_dir / output_filename)
                        
                    response = orchestrator.file_expert.respond(
                        user_input=instruction,
                        history=[],
                        memory=orchestrator.memory,
                        action=action_param,
                        file_path=str(target_path),
                        output_path=out_path,
                        instruction=instruction
                    )
                    
                    # Display response
                    st.markdown("#### Response Output:")
                    st.markdown(response)
                    
                    # If rewritten, provide download link
                    if action_param == "rewrite" and os.path.exists(out_path):
                        with open(out_path, "rb") as out_file:
                            st.download_button(
                                label="📥 Download Rewritten File",
                                data=out_file.read(),
                                file_name=output_filename,
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Failed to process file task: {e}")
