import os
import json
import uuid
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from agent.config import EMBEDDING_MODEL_NAME, get_api_key

class VectorStore:
    def __init__(self, filepath: str = "vector_store.json"):
        self.filepath = filepath
        self.documents = []  # List of dicts: {id, text, embedding, metadata}
        self.load()

    def load(self):
        """Loads vectors from JSON database."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading vector store: {e}. Starting fresh.")
                self.documents = []
        else:
            self.documents = []

    def save(self):
        """Saves vectors to JSON database."""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Calls Gemini API to get text embedding."""
        if not get_api_key():
            # Mock embedding for test/fallback mode if API key is not present
            # Return a deterministic mock vector based on hash or random
            # So tests can run without API key
            h = hash(text)
            np.random.seed(h % (2**32 - 1))
            return np.random.randn(768).tolist()

        try:
            # Clean text input (ensure string)
            text = str(text).strip()
            if not text:
                return [0.0] * 768
                
            response = genai.embed_content(
                model=EMBEDDING_MODEL_NAME,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            # Fallback to local deterministic mock vectors for testing if API call fails
            print(f"Embedding API call failed ({e}). Falling back to local mock embeddings.")
            h = hash(text)
            np.random.seed(h % (2**32 - 1))
            return np.random.randn(768).tolist()

    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Adds a document to the vector store with its embedding."""
        if metadata is None:
            metadata = {}
            
        doc_id = str(uuid.uuid4())
        embedding = self._get_embedding(text)
        
        doc = {
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
        
        self.documents.append(doc)
        self.save()
        return doc_id

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the store for most similar documents using cosine similarity."""
        if not self.documents:
            return []
            
        query_vector = self._get_embedding(query_text)
        q_vec = np.array(query_vector)
        
        results = []
        for doc in self.documents:
            d_vec = np.array(doc["embedding"])
            # Compute Cosine Similarity
            dot_product = np.dot(q_vec, d_vec)
            norm_q = np.linalg.norm(q_vec)
            norm_d = np.linalg.norm(d_vec)
            
            if norm_q == 0 or norm_d == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm_q * norm_d))
                
            # Create a return copy without embedding to save memory
            doc_copy = {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"]
            }
            results.append((doc_copy, similarity))
            
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self):
        """Clears the entire store."""
        self.documents = []
        self.save()
