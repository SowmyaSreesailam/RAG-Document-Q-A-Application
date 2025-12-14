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

## Project Structure

```
RAG_Application/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Document loading from various formats
│   ├── embedding.py         # Text chunking and embedding generation
│   ├── vectorstore.py       # FAISS vector store implementation
│   └── search.py            # RAG search and LLM integration
├── streamlit_app.py         # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (create this)
├── .gitignore
└── README.md
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### File Size Limits

- Maximum file size: 50MB (configurable in `streamlit_app.py`)

### Vector Store

- Default location: `faiss_store/`
- Index file: `faiss_store/faiss.index`
- Metadata file: `faiss_store/meta.pkl`

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

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure you have created a `.env` file with your API key
   - Check that the key is correctly formatted

2. **"No documents loaded"**
   - Verify file format is supported
   - Check file size is under the limit
   - Ensure files are not corrupted

3. **Import errors**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt` again

## License

This project is open source and available for personal and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the repository.

