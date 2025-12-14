# RAG Document Q&A Application

A modern Retrieval-Augmented Generation (RAG) application built with Streamlit that allows you to upload documents, index them, and ask questions using AI-powered search.

## Features

- 📄 **Multi-format Document Support**: Upload and process PDF, TXT, CSV, XLSX, DOCX, and JSON files
- 🔍 **Semantic Search**: Uses FAISS vector store with sentence transformers for efficient similarity search
- 🤖 **AI-Powered Answers**: Leverages OpenAI's GPT-4o-mini to generate contextual answers
- 🎨 **Modern UI**: Clean, subtle design with Inter font and intuitive interface
- 📚 **Source Tracking**: View source documents and retrieved context for each answer
- 💾 **Persistent Storage**: Vector store is saved locally for quick access

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG_Application
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. The application will open in your browser at `http://localhost:8501`

3. **Upload Documents**:
   - Use the sidebar to upload documents (PDF, TXT, CSV, XLSX, DOCX, JSON)
   - Click "📥 Index documents" to process and index them

4. **Ask Questions**:
   - Enter your question in the chat input at the bottom
   - The AI will search through your indexed documents and provide an answer
   - View sources and retrieved context by expanding the respective sections

5. **Clear Results**:
   - Use the 🗑️ button in the top right to clear current query and results



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

