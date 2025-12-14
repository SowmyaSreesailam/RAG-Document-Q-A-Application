import streamlit as st
import os
import base64
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.search import RAGSearch, FAISS_STORE_DIR

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
UPLOADED_DOCS_DIR = "uploaded_docs"
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.main-header {
    font-size: 2.4rem;
    font-weight: 600;
    text-align: center;
    color: #2d3748;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.answer-box {
    background-color: #f7fafc;
    padding: 1.4rem;
    border-radius: 12px;
    border-left: 4px solid #718096;
    color: #2d3748;
    line-height: 1.7;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.source-box {
    background-color: #f7fafc;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #a0aec0;
    margin-bottom: 0.8rem;
    color: #4a5568;
    line-height: 1.6;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.source-box:hover {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}

.source-box strong {
    color: #2d3748;
    font-weight: 600;
}

.source-box small {
    color: #718096;
    font-size: 0.9em;
}

h1, h2, h3 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-weight: 600;
    color: #2d3748;
    letter-spacing: -0.01em;
}

.stMarkdown {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Remove red hover colors and replace with subtle colors */
button:hover {
    border-color: #a0aec0 !important;
    color: #2d3748 !important;
}

button[kind="primary"]:hover {
    background-color: #718096 !important;
    border-color: #718096 !important;
}

button[kind="secondary"]:hover {
    background-color: #e2e8f0 !important;
    border-color: #a0aec0 !important;
    color: #2d3748 !important;
}

/* Remove red from any links or interactive elements */
a:hover {
    color: #718096 !important;
}

/* Override any Streamlit default red hover states */
.stButton > button:hover {
    border-color: #a0aec0 !important;
    color: #2d3748 !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #718096 !important;
    border-color: #718096 !important;
}

/* Style expander hover */
.stExpander:hover {
    border-color: #a0aec0 !important;
}

/* Remove red from any focus states */
button:focus,
input:focus,
textarea:focus,
select:focus {
    border-color: #a0aec0 !important;
    box-shadow: 0 0 0 2px rgba(160, 174, 192, 0.2) !important;
    outline: none !important;
}

/* Query input (chat input) - remove red hover/focus */
div[data-testid="stChatInputContainer"] > div > div:hover,
div[data-testid="stChatInputContainer"] > div > div:focus,
div[data-testid="stChatInputContainer"] input:hover,
div[data-testid="stChatInputContainer"] input:focus {
    border-color: #a0aec0 !important;
    box-shadow: 0 0 0 2px rgba(160, 174, 192, 0.2) !important;
}

div[data-testid="stChatInputContainer"] button:hover {
    background-color: #718096 !important;
    border-color: #718096 !important;
    color: white !important;
}

/* Source expander - remove red hover */
div[data-testid="stExpander"]:has(> details > summary:contains("📚")) summary:hover,
div[data-testid="stExpander"] summary:hover {
    color: #2d3748 !important;
    background-color: #f7fafc !important;
    border-color: #a0aec0 !important;
}

/* Retrieved Context expander - remove red hover */
div[data-testid="stExpander"]:has(> details > summary:contains("🔍")) summary:hover {
    color: #2d3748 !important;
    background-color: #f7fafc !important;
    border-color: #a0aec0 !important;
}

/* File uploader (Browse files button) - remove red hover */
div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] > div > button:hover {
    background-color: #718096 !important;
    border-color: #718096 !important;
    color: white !important;
}

div[data-testid="stFileUploader"] button:focus,
div[data-testid="stFileUploader"] > div > button:focus {
    border-color: #a0aec0 !important;
    box-shadow: 0 0 0 2px rgba(160, 174, 192, 0.2) !important;
}

/* All expander details hover */
details summary:hover {
    color: #2d3748 !important;
    background-color: #f7fafc !important;
}

/* Remove any red borders or outlines from interactive elements */
div[data-testid="stChatInputContainer"],
div[data-testid="stExpander"],
div[data-testid="stFileUploader"] {
    border-color: #e2e8f0 !important;
}

div[data-testid="stChatInputContainer"]:hover,
div[data-testid="stChatInputContainer"]:focus-within {
    border-color: #a0aec0 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Init RAG
# -----------------------------
if "rag" not in st.session_state:
    try:
        st.session_state.rag = RAGSearch()
        st.session_state.history = []
        st.session_state.last_query = None
        st.session_state.query_counter = 0
        st.session_state.current_result = None
    except ValueError as e:
        st.error(f"Initialization error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error during initialization: {str(e)}")
        logger.error(f"RAG initialization error: {str(e)}", exc_info=True)
        st.stop()

# -----------------------------
# PDF Preview
# -----------------------------
def show_pdf(file):
    pdf_base64 = base64.b64encode(file.getvalue()).decode()
    st.markdown(
        f"<iframe src='data:application/pdf;base64,{pdf_base64}' width='100%' height='300'></iframe>",
        unsafe_allow_html=True
    )

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Supported: PDF, TXT, CSV, XLSX, DOCX, JSON",
        type=["pdf", "txt", "csv", "xlsx", "docx", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(UPLOADED_DOCS_DIR, exist_ok=True)
        paths = []
        file_errors = []

        for f in uploaded_files:
            # Check file size
            file_size = len(f.getbuffer())
            if file_size > MAX_FILE_SIZE_BYTES:
                error_msg = f"❌ {f.name} exceeds {MAX_FILE_SIZE_MB}MB limit ({file_size / (1024*1024):.2f}MB)"
                st.warning(error_msg)
                file_errors.append(error_msg)
                continue
            
            try:
                path = os.path.join(UPLOADED_DOCS_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(path)
                st.caption(f"✔ {f.name} ({(file_size / (1024*1024)):.2f}MB)")
            except Exception as e:
                error_msg = f"❌ Error saving {f.name}: {str(e)}"
                st.error(error_msg)
                file_errors.append(error_msg)
                logger.error(error_msg, exc_info=True)

        if file_errors:
            with st.expander("⚠️ File Upload Errors", expanded=True):
                for error in file_errors:
                    st.text(error)

        if paths and st.button("📥 Index documents"):
            with st.spinner("Indexing documents..."):
                try:
                    result = st.session_state.rag.index_documents(paths)
                    
                    if result["success"]:
                        success_msg = f"✅ Successfully indexed {result['doc_count']} documents!"
                        if result["errors"]:
                            success_msg += f" ({len(result['errors'])} files had errors)"
                        st.success(success_msg)
                        
                        if result["errors"]:
                            with st.expander("⚠️ Indexing Errors", expanded=False):
                                for error in result["errors"]:
                                    st.text(error)
                        
                        # Clean up uploaded files after successful indexing
                        try:
                            for path in paths:
                                if os.path.exists(path):
                                    os.remove(path)
                            logger.info("Cleaned up uploaded files after indexing")
                        except Exception as e:
                            logger.warning(f"Failed to clean up files: {str(e)}")
                    else:
                        st.error("Failed to index documents")
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Unexpected error during indexing: {str(e)}")
                    logger.error(f"Indexing error: {str(e)}", exc_info=True)

        pdfs = [f for f in uploaded_files if f.type == "application/pdf"]
        if pdfs:
            st.markdown("### 📖 PDF Preview")
            try:
                show_pdf(pdfs[0])
            except Exception as e:
                st.warning(f"Could not preview PDF: {str(e)}")

    if st.button("🗑 Clear index"):
        try:
            if os.path.exists(FAISS_STORE_DIR):
                shutil.rmtree(FAISS_STORE_DIR, ignore_errors=True)
            st.session_state.rag = RAGSearch()
            st.success("✅ Index cleared successfully")
            logger.info("Vector store index cleared")
        except Exception as e:
            st.error(f"❌ Error clearing index: {str(e)}")
            logger.error(f"Error clearing index: {str(e)}", exc_info=True)

    st.markdown("---")
    st.header("🕘 Query History")
    for q in reversed(st.session_state.history[-6:]):
        st.caption(q)

# =============================
# MAIN AREA
# =============================
# Header with clear button on top right
col1, col2 = st.columns([10, 1])
with col1:
    st.markdown('<div class="main-header">🤖 RAG Document Q&A</div>', unsafe_allow_html=True)
with col2:
    if st.button("🗑️", help="Clear current query, answer, sources, and retrieved context", use_container_width=True):
        st.session_state.current_result = None
        st.session_state.last_query = None
        st.session_state.query_counter += 1
        st.rerun()

st.markdown("---")

# Query input
query = st.chat_input("Ask something about your documents...")

# Clear previous results if this is a new query
if query and query != st.session_state.get("last_query"):
    st.session_state.query_counter += 1
    st.session_state.last_query = query
    st.session_state.current_result = None  # Clear previous result

# Process query if provided
if query:
    with st.spinner("Searching..."):
        try:
            result = st.session_state.rag.search_with_details(query)
            st.session_state.history.append(query)
            st.session_state.current_result = result  # Store current result
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            logger.warning(f"Search error: {str(e)}")
            st.session_state.current_result = None
        except Exception as e:
            st.error(f"❌ Unexpected error during search: {str(e)}")
            logger.error(f"Search error: {str(e)}", exc_info=True)
            st.session_state.current_result = None

# Display results only for the current query
# When current_result is None (new query), nothing is displayed (clears previous results)
if st.session_state.get("current_result"):
    result = st.session_state.current_result
    current_query = st.session_state.get("last_query", "")
    
    # Display question
    if current_query:
        st.markdown("### ❓ Question")
        st.markdown(f"<div style='padding: 1rem; background-color: #f7fafc; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #a0aec0; color: #2d3748; line-height: 1.6; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);'><strong>{current_query}</strong></div>", unsafe_allow_html=True)
    
    # Display answer
    st.markdown("### 📝 Answer")
    st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add space between answer and sources

    # Display sources - hidden by default, shown when user clicks the button
    if result.get("sources"):
        with st.expander("📚 Sources", expanded=False):
            for s in result["sources"]:
                st.markdown(f"""
                <div class="source-box">
                <strong>Chunk {s['index']}</strong> (Score: {s['similarity']:.2f})<br>
                <small>{s['text']}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        with st.expander("📚 Sources", expanded=False):
            st.info("No sources found for this query.")

    # Display context - only shown when current_result exists (cleared on new query)
    if result.get("context"):
        with st.expander("🔍 Retrieved Context", expanded=False):
            st.text_area(
                "", 
                result["context"], 
                height=200, 
                disabled=True, 
                key=f"context_display_{st.session_state.query_counter}"
            )
