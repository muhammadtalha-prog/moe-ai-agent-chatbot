from typing import List, Dict, Any
from agent.config import get_groq_client, get_groq_key, GROQ_CHAT_MODEL_NAME
from agent.experts.base_expert import BaseExpert
from agent.memory import VectorStore

class ChatExpert(BaseExpert):
    def respond(self, user_input: str, history: List[Dict[str, str]], memory: VectorStore, **kwargs) -> str:
        """
        Responds to general chat using the Groq API, incorporating history.
        """
        if not get_groq_key():
            return "General Chat Expert: [API Key Not Configured] Hello! I'm in offline mock mode. Please configure GROQ_API_KEY in a .env file to enable fully functional AI responses."

        try:
            client = get_groq_client()
            
            # Convert internal message history to Groq format (roles: 'user', 'assistant', 'system')
            groq_messages = []
            
            # System prompt to keep responses concise and markdown friendly
            groq_messages.append({
                "role": "system",
                "content": "You are a helpful and intelligent AI chatbot coding assistant. Answer the user's questions clearly, concisely, and using Markdown."
            })
            
            # Add historical context
            for msg in history[-10:]:  # Keep recent history
                role = msg.get("role", "user")
                # Normalize assistant role name
                if role == "model":
                    role = "assistant"
                content = msg.get("content", "")
                if content:
                    groq_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Append the current prompt if not already the last message in history
            # (main.py appends it to history before calling the orchestrator, so check if it's there)
            if not groq_messages or groq_messages[-1]["content"] != user_input:
                groq_messages.append({
                    "role": "user",
                    "content": user_input
                })
            
            # Send message
            chat_completion = client.chat.completions.create(
                messages=groq_messages,
                model=GROQ_CHAT_MODEL_NAME,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"ChatExpert Error: {str(e)}"
