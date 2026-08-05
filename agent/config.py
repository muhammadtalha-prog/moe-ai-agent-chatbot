import os
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq
import time
import threading
from functools import wraps
from typing import Callable, Any
from collections import deque

# Resolve the absolute path of the project directory and load .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# SECURITY: API keys must be provided via .env or Streamlit secrets.
# Never hardcode API keys in source code.

def is_valid_key(key: str) -> bool:
    """Helper to verify if a key is a real key or just a placeholder."""
    if not key:
        return False
    key_stripped = key.strip()
    if not key_stripped or "your_" in key_stripped.lower() or "placeholder" in key_stripped.lower():
        return False
    return True

# 1. Gemini Configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL_NAME") or "models/gemini-embedding-001"

# 2. Groq Configurations
GROQ_CHAT_MODEL_NAME = os.getenv("GROQ_CHAT_MODEL") or os.getenv("GROQ_CHAT_MODEL_NAME") or "llama-3.3-70b-versatile"

def get_api_key() -> str:
    """Returns Gemini API key if configured via environment variables."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if is_valid_key(key):
        return key
    return None

def get_groq_client() -> Groq:
    """
    Returns an initialized Groq client instance.
    """
    key = get_groq_key()
    if not key:
        raise ValueError(
            "Groq API key is not configured. Please define GROQ_API_KEY in your environment or a .env file."
        )
    return Groq(api_key=key)

def get_groq_key() -> str:
    """Returns Groq API key if configured."""
    key = os.getenv("GROQ_API_KEY")
    if is_valid_key(key):
        return key
    return None

def get_embedding_model() -> Callable[..., Any]:
    """Get embedding model with proper fallback and error handling."""
    gemini_key = get_api_key()
    if not gemini_key:
        raise ValueError(
            "GEMINI_API_KEY is required for embeddings. "
            "Please set it in .env or Streamlit secrets."
        )
    try:
        genai.configure(api_key=gemini_key)
        return genai.embed_content
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini embeddings: {e}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[..., Any]:
    """Decorator to retry failed API calls with exponential backoff."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        from agent.security import SafeLogger
                        SafeLogger.log_error(e, f"API call failed after {max_retries} attempts in function '{func.__name__}'")
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

def rate_limit(calls_per_minute: int = 60) -> Callable[..., Any]:
    """Thread-safe rate limit decorator for API calls."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        last_called = [0.0]
        lock = threading.Lock()
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with lock:
                elapsed = time.time() - last_called[0]
                min_interval = 60.0 / calls_per_minute
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

