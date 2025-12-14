# RAG Document Q&A Application

A modern Retrieval-Augmented Generation (RAG) application built with Streamlit that allows you to upload documents, index them, and ask questions using AI-powered search.

## Features

- **Multi-format Document Support**: Upload and process PDF, TXT, CSV, XLSX, DOCX, and JSON files
- **Semantic Search**: Uses FAISS vector store with sentence transformers for efficient similarity search
- **AI-Powered Answers**: Leverages OpenAI's GPT-4o-mini to generate contextual answers
- **Modern UI**: Clean, subtle design with Inter font and intuitive interface
- **Source Tracking**: View source documents and retrieved context for each answer
- **Persistent Storage**: Vector store is saved locally for quick access

## Technologies Used

- **Streamlit**: Web application framework
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **LangChain**: Document processing and LLM integration
- **OpenAI**: GPT-4o-mini for answer generation
- **Python-dotenv**: Environment variable management

## Features in Detail

### Document Processing
- Automatic text extraction from multiple file formats
- Intelligent text chunking with overlap for better context
- Error handling for corrupted or unsupported files

### Vector Search
- Uses FAISS (Facebook AI Similarity Search) for fast vector operations
- Sentence transformer embeddings for semantic understanding
- Configurable number of top results (default: 5)

### Answer Generation
- Context-aware responses using retrieved documents
- Source attribution for transparency
- Handles cases where no relevant information is found

