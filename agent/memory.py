import os
import json
import uuid
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any, Tuple
from agent.config import EMBEDDING_MODEL_NAME, get_api_key
import logging
logger = logging.getLogger(__name__)

class VectorMemory:
    def __init__(self, filepath: str = "vector_store.json"):
        self.filepath = filepath
        self.embedding_version = "gemini-v1" if get_api_key() else "mock-v1"
        self.EMBEDDING_VERSION = self.embedding_version
        self.documents = []  # List of dicts: {id, text, embedding, metadata}
        self.embedding_cache = {}
        self.cache_filepath = "embedding_cache.json"
        self.load_cache()
        self.load()

    def load_cache(self):
        """Loads embedding cache from a JSON file."""
        if os.path.exists(self.cache_filepath):
            try:
                with open(self.cache_filepath, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.embedding_cache = {}
                try:
                    os.remove(self.cache_filepath)
                except Exception:
                    pass
            except Exception:
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}

    def save_cache(self):
        """Saves embedding cache to a JSON file."""
        try:
            with open(self.cache_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f, indent=2)
        except Exception as e:
            print(f"Error saving embedding cache: {e}")

    def load(self):
        """Loads vectors from JSON database, checking for format compatibility and embedding version."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Legacy structure: rebuild or migrate
                    print("Legacy vector store format detected. Migrating...")
                    self.migrate_or_rebuild(data)
                elif isinstance(data, dict):
                    meta = data.get('_metadata', {})
                    if meta.get('embedding_version') != self.embedding_version:
                        print(f"Embedding version mismatch ({meta.get('embedding_version')} != {self.embedding_version}). Rebuilding/migrating...")
                        self.migrate_or_rebuild(data.get('documents', []))
                    else:
                        self.documents = data.get('documents', [])
                else:
                    self.documents = []
            except Exception as e:
                print(f"Error loading vector store: {e}. Starting fresh.")
                self.documents = []
        else:
            self.documents = []

    def check_and_rebuild_store(self):
        """
        Check if vector store is valid for the current embedding model.
        Rebuild if invalid.
        """
        if not self.documents:
            return
        
        # Check a sample embedding for validity
        sample_doc = self.documents[0]
        sample_embedding = sample_doc.get('embedding', [])
        
        # Gemini embeddings are normalized unit vectors
        is_valid = self._validate_embedding(sample_embedding)
        
        if not is_valid:
            logger.warning("Invalid embedding format detected. Rebuilding vector store...")
            self.rebuild_all_embeddings()
            return
        
        # Check version
        version = sample_doc.get('embedding_version') or sample_doc.get('metadata', {}).get('embedding_version', 'unknown')
        if version != self.EMBEDDING_VERSION:
            logger.warning(f"Embedding version mismatch: {version} vs {self.EMBEDDING_VERSION}")
            self.rebuild_all_embeddings()

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that embedding format matches current model."""
        if not embedding or len(embedding) != 768:  # Gemini's dimension
            return False
        
        # Check normalization (Gemini embeddings are unit vectors)
        import math
        norm = math.sqrt(sum(x * x for x in embedding))
        return 0.99 <= norm <= 1.01

    @property
    def embedding_function(self):
        """Getter for embedding function used for rebuilds."""
        return self._get_embedding

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts using the embedding API or fallback to individual."""
        if get_api_key():
            try:
                from agent.config import get_embedding_model
                embed_fn = get_embedding_model()
                response = embed_fn(
                    model=EMBEDDING_MODEL_NAME,
                    content=texts,
                    task_type="retrieval_document",
                    output_dimensionality=768
                )
                return response['embedding']
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}. Falling back to individual.")
                return [self._get_embedding(t) for t in texts]
        else:
            return [self._get_embedding(t) for t in texts]

    def rebuild_all_embeddings(self):
        """Rebuild all embeddings using current embedding function."""
        if not self.embedding_function:
            logger.error("No embedding function available for rebuild")
            return
        
        if not self.documents:
            return
        
        texts = [doc['text'] for doc in self.documents]
        try:
            embeddings = self._batch_embed(texts)
            for doc, embedding in zip(self.documents, embeddings):
                doc['embedding'] = embedding
                doc['embedding_version'] = self.EMBEDDING_VERSION
                if 'metadata' not in doc or not isinstance(doc['metadata'], dict):
                    doc['metadata'] = {}
                doc['metadata']['embedding_version'] = self.EMBEDDING_VERSION
            self._save_store()
            logger.info(f"Rebuilt {len(self.documents)} embeddings")
        except Exception as e:
            logger.error(f"Failed to rebuild embeddings: {e}")

    def _save_store(self):
        """Save store helper alias."""
        self.save()

    def migrate_or_rebuild(self, legacy_documents: List[Dict[str, Any]]):
        """Migrate legacy documents to new embedding version if possible, otherwise clear."""
        if not legacy_documents:
            self.documents = []
            self.save()
            return
            
        print(f"Migrating vector store to embedding version '{self.embedding_version}'...")
        self.documents = []
        
        texts = []
        metadatas = []
        for doc in legacy_documents:
            if isinstance(doc, dict) and 'text' in doc:
                texts.append(doc['text'])
                meta = doc.get('metadata') or {}
                meta['embedding_version'] = self.embedding_version
                metadatas.append(meta)
                
        if texts:
            try:
                self.add_documents_batch(texts, metadatas)
            except Exception as e:
                print(f"Migration failed: {e}. Starting with empty vector store.")
                self.documents = []
                self.save()
        else:
            self.save()

    def save(self):
        """Saves vectors to JSON database with metadata wrapper."""
        try:
            data = {
                "_metadata": {
                    "embedding_version": self.embedding_version
                },
                "documents": self.documents
            }
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Calls Gemini API to get text embedding with local caching."""
        text_clean = str(text).strip()
        if not text_clean:
            return [0.0] * 768
            
        if text_clean in self.embedding_cache:
            return self.embedding_cache[text_clean]

        if not get_api_key():
            # Mock embedding for test/fallback mode if API key is not present
            h = hash(text_clean)
            np.random.seed(h % (2**32 - 1))
            val = np.random.randn(768).tolist()
            self.embedding_cache[text_clean] = val
            self.save_cache()
            return val

        try:
            from agent.config import get_embedding_model
            embed_fn = get_embedding_model()
            response = embed_fn(
                model=EMBEDDING_MODEL_NAME,
                content=text_clean,
                task_type="retrieval_document",
                output_dimensionality=768
            )
            val = response['embedding']
            self.embedding_cache[text_clean] = val
            self.save_cache()
            return val
        except Exception as e:
            # Fallback to local mock embeddings if API call fails
            print(f"Embedding API call failed ({e}). Falling back to local mock embeddings.")
            h = hash(text_clean)
            np.random.seed(h % (2**32 - 1))
            val = np.random.randn(768).tolist()
            self.embedding_cache[text_clean] = val
            self.save_cache()
            return val

    def add_text(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Adds a document to the vector store with its embedding."""
        if metadata is None:
            metadata = {}
            
        metadata['embedding_version'] = self.embedding_version
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

    def add_documents_batch(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[str]:
        """Add multiple documents in batch for embedding and save to database."""
        if not texts:
            return []
            
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
            
        embeddings = []
        if get_api_key():
            try:
                from agent.config import get_embedding_model
                embed_fn = get_embedding_model()
                # Try batching via api if supported, otherwise loop with _get_embedding which has cache checks
                response = embed_fn(
                    model=EMBEDDING_MODEL_NAME,
                    content=texts,
                    task_type="retrieval_document"
                )
                embeddings = response['embedding']
                for txt, emb in zip(texts, embeddings):
                    self.embedding_cache[str(txt).strip()] = emb
                self.save_cache()
            except Exception as e:
                print(f"Batch embedding failed ({e}). Falling back to individual cached embeddings.")
                embeddings = [self._get_embedding(text) for text in texts]
        else:
            embeddings = [self._get_embedding(text) for text in texts]
            
        doc_ids = []
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            doc_id = str(uuid.uuid4())
            if not isinstance(metadata, dict):
                metadata = {}
            metadata['embedding_version'] = self.embedding_version
            
            doc = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            }
            self.documents.append(doc)
            doc_ids.append(doc_id)
            
        self.save()
        return doc_ids

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Dict[str, Any], float]]:
        """Queries the store for most similar documents using cosine similarity."""
        if not self.documents:
            return []
            
        query_vector = self._get_embedding(query_text)
        q_vec = np.array(query_vector)
        
        results = []
        for doc in self.documents:
            # Skip legacy error records to avoid context pollution
            text_lower = doc["text"].lower()
            if "chatexpert error" in text_lower or "error code: 401" in text_lower or "invalid_api_key" in text_lower or "fileexpert error" in text_lower:
                continue
                
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

VectorStore = VectorMemory
