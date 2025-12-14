import logging
from typing import List, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the embedding pipeline.
        
        Args:
            model: SentenceTransformer model name
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
        """
        self.model = SentenceTransformer(model)
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"Initialized EmbeddingPipeline with model: {model}")

    def chunk(self, docs: List[Any]) -> List[Any]:
        """
        Split documents into chunks.
        
        Args:
            docs: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        if not docs:
            logger.warning("Empty document list provided for chunking")
            return []
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks")
        return chunks

    def embed(self, chunks: List[Any]) -> np.ndarray:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Numpy array of embeddings
        """
        if not chunks:
            logger.warning("Empty chunk list provided for embedding")
            return np.array([])
        
        try:
            texts = [c.page_content for c in chunks]
            emb = self.model.encode(texts).astype("float32")
            logger.debug(f"Generated embeddings for {len(chunks)} chunks")
            return emb
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise
