import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.vectorstore import FaissVectorStore
from src.data_loader import load_uploaded_documents

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration constants
FAISS_STORE_DIR = "faiss_store"

class RAGSearch:
    def __init__(self, faiss_store_dir: str = FAISS_STORE_DIR):
        """Initialize RAG search with vector store and LLM."""
        self.store = FaissVectorStore(persist_dir=faiss_store_dir)

        # Try to load existing index
        index_path = os.path.join(faiss_store_dir, "faiss.index")
        if os.path.exists(index_path):
            try:
                self.store.load()
                logger.info("Loaded existing vector store")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")

        # Validate and initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model="gpt-4o-mini"
            )
            logger.info("Initialized ChatOpenAI with gpt-4o-mini")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise

    def index_documents(self, paths: List[str]) -> Dict[str, Any]:
        """
        Index documents from file paths.
        
        Returns:
            Dictionary with 'success' (bool), 'doc_count' (int), and 'errors' (list)
        """
        if not paths:
            raise ValueError("No file paths provided")
        
        try:
            docs, errors = load_uploaded_documents(paths)
            
            if not docs:
                error_msg = "No documents loaded from provided paths"
                if errors:
                    error_msg += f". Errors: {', '.join(errors)}"
                raise ValueError(error_msg)
            
            self.store.add_documents(docs)
            
            result = {
                "success": True,
                "doc_count": len(docs),
                "errors": errors
            }
            
            if errors:
                logger.warning(f"Indexed {len(docs)} documents with {len(errors)} errors")
            else:
                logger.info(f"Successfully indexed {len(docs)} documents")
            
            return result
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}", exc_info=True)
            raise

    def search_with_details(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for relevant documents and generate an answer.
        
        Args:
            query: The search query
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with 'answer', 'sources', and 'context'
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            results = self.store.query(query, top_k=top_k)
            
            if not results:
                logger.warning("No results found for query")
                return {
                    "answer": "I couldn't find any relevant information in the indexed documents to answer your question.",
                    "sources": [],
                    "context": ""
                }
            
            context = "\n\n".join([r["text"] for r in results])
            
            if not context.strip():
                logger.warning("Empty context generated from results")
                return {
                    "answer": "I found some results but couldn't extract meaningful context.",
                    "sources": [],
                    "context": ""
                }

            messages = [
                {
                    "role": "system",
                    "content": "Answer only using the provided context. If the context doesn't contain enough information to answer the question, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                }
            ]

            try:
                response = self.llm.invoke(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"LLM API error: {str(e)}", exc_info=True)
                raise ValueError(f"Failed to generate answer: {str(e)}")

            sources = [
                {
                    "index": i + 1,
                    "text": r["text"][:200] + ("..." if len(r["text"]) > 200 else ""),
                    "similarity": r["score"]
                }
                for i, r in enumerate(results)
            ]

            logger.info(f"Generated answer for query: {query[:50]}...")
            return {
                "answer": answer,
                "sources": sources,
                "context": context
            }
        except Exception as e:
            logger.error(f"Error in search_with_details: {str(e)}", exc_info=True)
            raise
