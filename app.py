import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Set page configurations first before any other Streamlit commands
st.set_page_config(
    page_title="AI Agent Chatbot (MoE)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to check if key is valid (not placeholder)
def is_placeholder_key(key: str) -> bool:
    if not key:
        return True
    key_stripped = key.strip()
    return not key_stripped or "your_" in key_stripped.lower() or "placeholder" in key_stripped.lower()

# Get initial defaults
default_groq = st.session_state.get("custom_groq_key", os.getenv("GROQ_API_KEY", ""))
default_gemini = st.session_state.get("custom_gemini_key", os.getenv("GEMINI_API_KEY", ""))

if is_placeholder_key(default_groq):
    default_groq = ""
if is_placeholder_key(default_gemini):
    default_gemini = ""

# API Key Dynamic Inputs (rendered in sidebar but evaluated first)
st.sidebar.subheader("🛠️ Configure API Keys")
custom_groq = st.sidebar.text_input(
    "Groq API Key (gsk_...)",
    value=default_groq,
    type="password",
    help="Enter your Groq API Key. If left empty, the app will try to read GROQ_API_KEY from environment or secrets."
)
custom_gemini = st.sidebar.text_input(
    "Gemini API Key (AIzaSy...)",
    value=default_gemini,
    type="password",
    help="Enter your Gemini API Key. If left empty, the app will try to read GEMINI_API_KEY from environment or secrets."
)

# Apply inputs to environment dynamically
if custom_groq:
    os.environ["GROQ_API_KEY"] = custom_groq
    st.session_state["custom_groq_key"] = custom_groq
else:
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    if "custom_groq_key" in st.session_state:
        st.session_state.pop("custom_groq_key")
        
if custom_gemini:
    os.environ["GEMINI_API_KEY"] = custom_gemini
    st.session_state["custom_gemini_key"] = custom_gemini
else:
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "custom_gemini_key" in st.session_state:
        st.session_state.pop("custom_gemini_key")

# Inject Streamlit Secrets / dotenv fallback if custom key is not present
if not os.environ.get("GROQ_API_KEY"):
    if "GROQ_API_KEY" in st.secrets and not is_placeholder_key(st.secrets["GROQ_API_KEY"]):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if not os.environ.get("GEMINI_API_KEY"):
    if "GEMINI_API_KEY" in st.secrets and not is_placeholder_key(st.secrets["GEMINI_API_KEY"]):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# Load dotenv fallback using absolute path
base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(base_dir, ".env"))

# Check if environment keys are still placeholders and clear them if so
if is_placeholder_key(os.environ.get("GROQ_API_KEY")):
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
if is_placeholder_key(os.environ.get("GEMINI_API_KEY")):
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

from agent.config import get_groq_key, get_api_key, GROQ_CHAT_MODEL_NAME
from agent.experts.orchestrator import Orchestrator
from agent.file_handler import read_file, write_file
from agent.security import get_workspace_dir

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

# Verification Cache Helper
def validate_groq_key(key: str) -> bool:
    if not key:
        return False
    import hashlib
    hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
    cache_key = f"val_groq_{hashed_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        from groq import Groq
        client = Groq(api_key=key)
        client.models.list()
        st.session_state[cache_key] = True
        return True
    except Exception:
        st.session_state[cache_key] = False
        return False

def validate_gemini_key(key: str) -> bool:
    if not key:
        return False
    import hashlib
    hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
    cache_key = f"val_gemini_{hashed_key}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        genai.list_models()
        st.session_state[cache_key] = True
        return True
    except Exception:
        st.session_state[cache_key] = False
        return False

# Evaluate dynamic keys
resolved_groq_key = get_groq_key()
resolved_gemini_key = get_api_key()

groq_valid = validate_groq_key(resolved_groq_key)
gemini_valid = validate_gemini_key(resolved_gemini_key)

# If any key is set but not valid, clear it from env so backend uses fallback mock mode
if resolved_groq_key and not groq_valid:
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    resolved_groq_key = None

if resolved_gemini_key and not gemini_valid:
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    resolved_gemini_key = None

# Initialize Orchestrator
def initialize_embedding_cache():
    cache_path = Path("embedding_cache.json")
    if cache_path.exists():
        # Check if cache is compatible
        import json
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache_path.unlink(missing_ok=True)
            return

        try:
            if data:
                sample_key = next(iter(data.keys()))
                sample_embedding = data[sample_key]
                
                # Check validation
                is_valid = False
                if sample_embedding and len(sample_embedding) == 768:
                    import math
                    norm = math.sqrt(sum(x * x for x in sample_embedding))
                    is_gemini = (0.99 <= norm <= 1.01)
                    
                    from agent.config import get_api_key
                    if get_api_key():
                        is_valid = is_gemini
                    else:
                        is_valid = not is_gemini
                        
                if not is_valid:
                    # Incompatible, delete cache
                    cache_path.unlink()
                    import logging
                    logging.getLogger("app").info("Deleted incompatible embedding cache")
        except Exception:
            pass

@st.cache_resource
def get_orchestrator(groq_key, gemini_key):
    """Cache orchestrator across reruns, re-initialize if keys change"""
    initialize_embedding_cache()
    orchestrator = Orchestrator(vector_store_path="vector_store.json")
    # Check and rebuild if needed
    orchestrator.memory.check_and_rebuild_store()
    return orchestrator

orchestrator = get_orchestrator(resolved_groq_key, resolved_gemini_key)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "max_history" not in st.session_state:
    st.session_state.max_history = 50

def trim_history():
    """Trim session state chat history to prevent memory leaks."""
    if len(st.session_state.chat_history) > st.session_state.max_history:
        st.session_state.chat_history = st.session_state.chat_history[-st.session_state.max_history:]
    if len(st.session_state.messages) > st.session_state.max_history:
        st.session_state.messages = st.session_state.messages[-st.session_state.max_history:]

# --- SIDEBAR PANELS ---
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/bot.png", width=80)
    st.title("AI Agent Settings")
    st.caption("Mixture of Experts | MCP | Semantic Memory")
    st.markdown("---")
    
    # API Key status panel
    st.subheader("🔑 API Configurations")
    
    if resolved_groq_key:
        st.success("🟢 Groq API: Connected")
        st.info(f"Model: {GROQ_CHAT_MODEL_NAME}")
    else:
        st.error("🔴 Groq API: Offline Mock Mode")
        st.warning("No valid Groq API key configured. AI responses will use fallback offline mocks.")
        
    if resolved_gemini_key:
        st.success("🟢 Gemini API: Connected (Embeddings)")
    else:
        st.warning("🟡 Gemini API: Offline Fallback (Mock Embeddings Active)")
        
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
    with col2:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            
    st.markdown("---")
    
    # System health check panel
    st.subheader("⚙️ System Status")
    
    def check_health():
        """Check all system components"""
        status = {
            "status": "healthy",
            "components": {}
        }
        
        # Check API keys
        status["components"]["groq"] = "connected" if get_groq_key() else "missing"
        status["components"]["gemini"] = "connected" if get_api_key() else "missing"
        
        # Check vector store
        try:
            status["components"]["vector_store"] = {
                "status": "accessible",
                "records": len(orchestrator.memory.documents),
                "version": orchestrator.memory.embedding_version
            }
        except Exception as e:
            status["components"]["vector_store"] = {
                "status": f"corrupted or missing ({e})"
            }
            status["status"] = "degraded"
        
        # Check workspace directories
        try:
            workspace = get_workspace_dir()
            temp_dir = workspace / "uploaded_files"
            status["components"]["workspace"] = {
                "status": "accessible",
                "path": str(workspace),
                "uploads_exist": temp_dir.exists()
            }
        except Exception as e:
            status["components"]["workspace"] = {
                "status": f"inaccessible ({e})"
            }
            status["status"] = "degraded"
            
        return status

    if st.button("🔍 Run System Health Check", use_container_width=True):
        health = check_health()
        if health["status"] == "healthy":
            st.success("System Status: HEALTHY")
        else:
            st.warning("System Status: DEGRADED")
        st.json(health)

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
        from agent.security import sanitize_user_input
        user_query = sanitize_user_input(user_query)
        
        # Append User Input
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        trim_history()
        
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
                    trim_history()
                except Exception as e:
                    from agent.security import SafeLogger
                    SafeLogger.log_error(e, "Error executing chat query")
                    st.error(f"Error executing chat: {e}")

# Tab 2: File upload processing
with tab2:
    st.markdown("### 📁 File Processing Expert")
    st.caption("Upload files in any format (e.g. Word, Excel, PDF, CSV, JSON, HTML, YAML, etc.). Extract summaries or rewrite content safely in the same file format.")

    uploaded_file = st.file_uploader(
        "Choose a file to analyze"
    )
    
    validated_file = False
    if uploaded_file is not None:
        try:
            from agent.security import validate_and_secure_file
            filename, file_bytes = validate_and_secure_file(uploaded_file)
            
            # Safe storage in the workspace folder
            workspace = get_workspace_dir()
            temp_dir = workspace / "uploaded_files"
            temp_dir.mkdir(exist_ok=True)
            
            # Resolve target path and verify traversal safety
            from agent.security import get_safe_path
            target_path = get_safe_path(temp_dir / filename, workspace)
            
            # Write bytes safely to disk
            with open(target_path, "wb") as f:
                f.write(file_bytes)
                
            st.success(f"Uploaded and secured file: `{filename}`")
            validated_file = True
        except Exception as e:
            from agent.security import SafeLogger
            SafeLogger.log_error(e, "File upload security validation failed")
            st.error(f"File validation error: {e}")
            
    if uploaded_file is not None and validated_file:
        # Description section (prompt input)
        instruction = st.text_area(
            "What would you like to do with this file?",
            placeholder="e.g. 'summarize this document', 'analyze readability and sentiment', 'translate to Spanish', 'rewrite to add comments'",
            help="Describe your request. The system will automatically determine the best action (Read, Summarize, Rewrite, Analyze, or Compare) based on your description."
        )
        
        # Optional second file uploader for comparison
        compare_file = st.file_uploader(
            "Upload a second file (Optional - for comparison requests)",
            key="compare_file_uploader"
        )
        target_compare_path = None
        if compare_file is not None:
            try:
                from agent.security import validate_and_secure_file
                compare_filename, compare_bytes = validate_and_secure_file(compare_file)
                workspace = get_workspace_dir()
                target_compare_path = get_safe_path(temp_dir / compare_filename, workspace)
                with open(target_compare_path, "wb") as f:
                    f.write(compare_bytes)
                st.success(f"Uploaded second file: `{compare_filename}`")
            except Exception as e:
                st.error(f"Second file validation error: {e}")

        if st.button("🔥 Execute File Action", type="primary"):
            with st.spinner("Processing file..."):
                try:
                    # Automatically classify user intent/action based on instruction and uploads
                    instr_lower = instruction.lower().strip()
                    action_param = ""
                    out_path = None
                    output_filename = ""
                    
                    if not instr_lower:
                        if compare_file is not None and target_compare_path is not None:
                            action_param = "compare"
                            out_path = str(target_compare_path)
                        else:
                            action_param = "read"
                    elif any(w in instr_lower for w in ["compare", "diff"]) or compare_file is not None:
                        action_param = "compare"
                        if compare_file is None or target_compare_path is None:
                            st.error("Please upload a second file to perform the comparison.")
                            st.stop()
                        out_path = str(target_compare_path)
                    elif any(w in instr_lower for w in ["analyze", "analysis", "audit", "entity", "entities", "sentiment", "readability"]):
                        action_param = "analyze"
                    elif any(w in instr_lower for w in ["rewrite", "modify", "edit", "change", "translate", "refactor", "update", "format", "add", "remove", "replace", "convert"]):
                        action_param = "rewrite"
                        base, ext = os.path.splitext(filename)
                        output_filename = f"{base}_modified{ext}"
                        out_path = str(temp_dir / output_filename)
                    else:
                        action_param = "summarize"
                        
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
                    if action_param == "rewrite" and out_path and os.path.exists(out_path):
                        with open(out_path, "rb") as out_file:
                            st.download_button(
                                label="📥 Download Rewritten File",
                                data=out_file.read(),
                                file_name=output_filename or f"modified_{filename}",
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"Failed to process file task: {e}")
