import os
import google.generativeai as genai
from dotenv import load_dotenv
from groq import Groq

# Load .env file if it exists
load_dotenv()

# Gemini Configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def get_api_key():
    """Returns Gemini API key if configured."""
    return GEMINI_API_KEY

# Groq Configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL_NAME = "llama-3.3-70b-versatile"

def get_groq_client() -> Groq:
    """
    Returns an initialized Groq client instance.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key is not configured. Please define GROQ_API_KEY in your environment or a .env file."
        )
    return Groq(api_key=GROQ_API_KEY)

def get_groq_key() -> str:
    """Returns Groq API key if configured."""
    return GROQ_API_KEY
