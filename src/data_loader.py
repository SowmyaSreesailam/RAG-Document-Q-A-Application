import logging
import os
from pathlib import Path
from typing import List, Any, Tuple
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

logger = logging.getLogger(__name__)

def load_uploaded_documents(paths: List[str]) -> Tuple[List[Any], List[str]]:
    """
    Load documents from file paths with error handling.
    
    Returns:
        Tuple of (documents, error_messages) where error_messages contains
        any files that failed to load.
    """
    documents = []
    errors = []

    for p in paths:
        if not os.path.exists(p):
            error_msg = f"File not found: {p}"
            logger.warning(error_msg)
            errors.append(error_msg)
            continue

        ext = Path(p).suffix.lower()
        loader = None

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(p)
            elif ext == ".txt":
                loader = TextLoader(p)
            elif ext == ".csv":
                loader = CSVLoader(p)
            elif ext in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(p)
            elif ext == ".docx":
                loader = Docx2txtLoader(p)
            elif ext == ".json":
                loader = JSONLoader(p, jq_schema=".", text_content=False)
            else:
                error_msg = f"Unsupported file type: {ext} for {p}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

            if loader:
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Successfully loaded {len(loaded_docs)} documents from {p}")

        except Exception as e:
            error_msg = f"Error loading {p}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

    return documents, errors
