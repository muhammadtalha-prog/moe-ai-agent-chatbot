import os
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

# Resolve the absolute path of the project directory and load .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

def is_valid_key(key: str) -> bool:
    """Helper to verify if a key is a real key or just a placeholder."""
    if not key:
        return False
    key_stripped = key.strip()
    if not key_stripped or "your_" in key_stripped.lower() or "placeholder" in key_stripped.lower():
        return False
    return True

# 1. Gemini Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# 2. Groq Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL_NAME = "llama-3.3-70b-versatile"

# 3. Fallback parser: If running on cloud and keys are missing/invalid, check .env.example
if not is_valid_key(GROQ_API_KEY) or not is_valid_key(GEMINI_API_KEY):
    example_path = os.path.join(base_dir, ".env.example")
    if os.path.exists(example_path):
        try:
            with open(example_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_stripped = line.strip()
                    if "=" in line_stripped and not line_stripped.startswith("#"):
                        k, v = line_stripped.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k == "GROQ_API_KEY" and not is_valid_key(GROQ_API_KEY):
                            GROQ_API_KEY = v
                        elif k == "GEMINI_API_KEY" and not is_valid_key(GEMINI_API_KEY):
                            GEMINI_API_KEY = v
        except Exception:
            pass

# Configure Gemini if key was successfully resolved
if is_valid_key(GEMINI_API_KEY):
    genai.configure(api_key=GEMINI_API_KEY)

def get_api_key():
    """Returns Gemini API key if configured."""
    return GEMINI_API_KEY if is_valid_key(GEMINI_API_KEY) else None

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
    return GROQ_API_KEY if is_valid_key(GROQ_API_KEY) else None
