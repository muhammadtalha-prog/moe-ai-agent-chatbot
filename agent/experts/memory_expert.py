from typing import List, Dict, Any
from agent.experts.base_expert import BaseExpert
from agent.memory import VectorStore

class MemoryExpert(BaseExpert):
    def respond(self, user_input: str, history: List[Dict[str, str]], memory: VectorStore, **kwargs) -> str:
        """
        Handles explicit memory operations (remember, recall, search).
        Kwargs:
          - action: 'remember' (write to vector memory) or 'recall' (query vector memory)
          - text_to_remember: Optional text to write to store
          - query_text: Optional text to search
        """
        action = kwargs.get("action")
        text_to_remember = kwargs.get("text_to_remember")
        query_text = kwargs.get("query_text") or user_input
        
        if action == 'remember':
            if not text_to_remember:
                return "MemoryExpert Error: Please provide the text you want me to remember."
            
            doc_id = memory.add_text(
                text=text_to_remember, 
                metadata={"source": "user_remember"}
            )
            return f"Understood! I have saved that to my memory. (ID: {doc_id[:8]})"
            
        elif action == 'recall' or action == 'search':
            if not query_text:
                return "MemoryExpert Error: Please provide a search query."
                
            results = memory.query(query_text, top_k=4)
            if not results:
                return "I couldn't find any relevant memories matching your query."
                
            response = ["### Relevant Memories Found:\n"]
            for i, (doc, score) in enumerate(results, 1):
                meta_str = f" [Source: {doc['metadata'].get('source', 'unknown')}]"
                response.append(f"{i}. **{doc['text']}** (Confidence: {score:.2f}){meta_str}")
                
            return "\n".join(response)
            
        else:
            # Default to querying memory
            results = memory.query(query_text, top_k=3)
            if not results:
                return "No matching memories found."
            
            response = ["Here is what I retrieved from memory:"]
            for doc, score in results:
                response.append(f"- {doc['text']} (Similarity: {score:.2f})")
            return "\n".join(response)
