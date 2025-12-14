import os
import logging
import faiss
import pickle
import numpy as np
from typing import List, Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

logger = logging.getLogger(__name__)

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

        # Use pipeline's model consistently
        self.pipeline = EmbeddingPipeline(model)
        self.model = self.pipeline.model  # Use the same model instance

    def add_documents(self, docs: List[Any]) -> None:
        """Add documents to the vector store."""
        if not docs:
            raise ValueError("Cannot add empty document list")
        
        try:
            chunks = self.pipeline.chunk(docs)
            if not chunks:
                raise ValueError("No chunks created from documents")
            
            emb = self.pipeline.embed(chunks)
            if emb.shape[0] == 0:
                raise ValueError("No embeddings generated")

            faiss.normalize_L2(emb)

            if self.index is None:
                self.index = faiss.IndexFlatIP(emb.shape[1])
                logger.info(f"Created new FAISS index with dimension {emb.shape[1]}")

            self.index.add(emb)
            self.metadata.extend([{"text": c.page_content} for c in chunks])
            self.save()
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise

    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self.index is None:
            raise ValueError("Cannot save: index is None")
        
        try:
            index_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "meta.pkl")
            
            faiss.write_index(self.index, index_path)
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved vector store to {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}", exc_info=True)
            raise

    def load(self) -> None:
        """Load the index and metadata from disk."""
        index_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "meta.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        try:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded vector store from {self.persist_dir} ({len(self.metadata)} chunks)")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}", exc_info=True)
            raise

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store for similar documents."""
        if self.index is None:
            raise ValueError("Index not initialized. Please load or add documents first.")
        
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        
        # Check if index is empty
        if self.index.ntotal == 0:
            logger.warning("Query attempted on empty index")
            return []
        
        try:
            # Use pipeline's model for consistency
            q = self.model.encode([text]).astype("float32")
            faiss.normalize_L2(q)
            
            # Ensure top_k doesn't exceed available documents
            actual_top_k = min(top_k, self.index.ntotal)
            D, I = self.index.search(q, actual_top_k)

            results = [
                {
                    "score": float(D[0][i]),
                    "text": self.metadata[idx]["text"]
                }
                for i, idx in enumerate(I[0]) if idx < len(self.metadata)
            ]
            
            logger.debug(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}", exc_info=True)
            raise
