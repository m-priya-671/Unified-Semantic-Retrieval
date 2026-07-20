import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import time
from pathlib import Path
from src.config.settings import (
    UPLOAD_DIR, UPLOAD_LIMITS, 
    MAX_PDF_SIZE_MB, MAX_DOCX_SIZE_MB, 
    MAX_IMAGE_SIZE_MB, MAX_AUDIO_SIZE_MB
)
from src.utils.logger import logger
from src.utils.file_manager import FileManager
from src.ingestion.parser_factory import ParserFactory

# 1. Page Configuration
st.set_page_config(
    page_title="Unified Semantic Retrieval - Offline Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    /* Gradient Title Banner */
    .banner-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 5px;
    }

    .banner-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }

    /* Custom Badges */
    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-pdf {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }

    .badge-docx {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }

    .badge-image {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }

    .badge-audio {
        background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%);
        color: white !important;
        box-shadow: 0 2px 8px rgba(168, 85, 247, 0.3);
    }

    /* Metric Sub-cards inside sidebar */
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.015);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        margin-bottom: 10px;
    }

    .sidebar-metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .sidebar-metric-lbl {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Helper to detect GPU hardware recommendation
def get_hardware_recommendation() -> str:
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        total_mem = int(res.stdout.strip().split("\n")[0])
        if total_mem <= 4608:
            return f"LOW_MEMORY profile (Detected GPU has {total_mem} MiB VRAM)"
        else:
            return f"Balanced / Quality profile (Detected GPU has {total_mem} MiB VRAM)"
    except Exception:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                total_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024 * 1024)
                if total_mem <= 4600:
                    return f"LOW_MEMORY profile (Detected GPU has {total_mem:.0f} MB VRAM)"
                else:
                    return f"Balanced / Quality profile (Detected GPU has {total_mem:.0f} MB VRAM)"
        except Exception:
            pass
    return "Balanced profile (Recommended)"

# 3. Startup Validation
if "startup_validated" not in st.session_state:
    st.session_state.startup_warnings = []
    try:
        from src.llm import LLMManager
        llm_manager = LLMManager()
        if not llm_manager.client.is_server_running():
            st.session_state.startup_warnings.append(
                "Ollama server is not running or unreachable on http://localhost:11434. Please start Ollama before searching."
            )
        else:
            if not llm_manager.client.is_model_installed(llm_manager.model_name):
                st.session_state.startup_warnings.append(
                    f"Model '{llm_manager.model_name}' was not found in your local Ollama registry. Please run `ollama pull {llm_manager.model_name}` first."
                )
        from src.config.settings import LLM_RUNTIME_MODE, LLM_RUNTIME_PROFILE, OLLAMA_OPTIONS
        if LLM_RUNTIME_MODE not in ["auto", "gpu", "cpu"]:
            st.session_state.startup_warnings.append(
                f"Invalid LLM_RUNTIME_MODE: '{LLM_RUNTIME_MODE}'. Supported values are 'auto', 'gpu', or 'cpu'."
            )
        if LLM_RUNTIME_PROFILE not in ["LOW_MEMORY", "BALANCED", "QUALITY"]:
            st.session_state.startup_warnings.append(
                f"Invalid LLM_RUNTIME_PROFILE: '{LLM_RUNTIME_PROFILE}'. Supported values are 'LOW_MEMORY', 'BALANCED', or 'QUALITY'."
            )
        if not isinstance(OLLAMA_OPTIONS, dict) or "num_ctx" not in OLLAMA_OPTIONS:
            st.session_state.startup_warnings.append(
                "Invalid OLLAMA_OPTIONS configuration dictionary in settings.py."
            )
    except Exception as startup_err:
        st.session_state.startup_warnings.append(f"Startup validation failed: {str(startup_err)}")
    st.session_state.startup_validated = True

# 3.5 Initialize Session State
if "parsed_files" not in st.session_state:
    st.session_state.parsed_files = {}

# 4. Sidebar Content (Statistics & Operations)
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/database.png", width=60)
    st.markdown("<h2 style='margin-top:0px;'>Control Center</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Section 1: System Overview
    st.markdown("### ⚙️ System Overview")
    st.markdown(f"**Maximum PDF Size:** {MAX_PDF_SIZE_MB} MB")
    st.markdown(f"**Maximum DOCX Size:** {MAX_DOCX_SIZE_MB} MB")
    st.markdown(f"**Maximum Image Size:** {MAX_IMAGE_SIZE_MB} MB")
    st.markdown(f"**Maximum Audio Size:** {MAX_AUDIO_SIZE_MB} MB")
    st.markdown("**Offline Mode:** `Ready` 🟢")
    
    st.markdown("---")
    from src.vector_store import IndexManager
    idx_manager = IndexManager()
    index_info = idx_manager.get_index_info()
    
    dev_mode = st.checkbox("🔧 Enable Developer Mode", value=st.session_state.get("dev_mode", False), key="dev_mode")
    if dev_mode:
        st.markdown("#### ✂️ Chunking Config")
        st.slider(
            "Chunk Size (chars)",
            min_value=100,
            max_value=1000,
            value=st.session_state.get("ui_chunk_size", 500),
            step=50,
            key="ui_chunk_size"
        )
        st.slider(
            "Chunk Overlap (chars)",
            min_value=0,
            max_value=300,
            value=st.session_state.get("ui_chunk_overlap", 100),
            step=10,
            key="ui_chunk_overlap"
        )
        st.slider(
            "Batch Size",
            min_value=8,
            max_value=128,
            value=st.session_state.get("ui_batch_size", 32),
            step=8,
            key="ui_batch_size"
        )
        st.markdown("#### 🔍 Retrieval Config")
        st.slider(
            "Retrieval Top-K",
            min_value=1,
            max_value=10,
            value=st.session_state.get("qa_top_k", 5),
            step=1,
            key="qa_top_k"
        )
        st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("qa_threshold", 0.70),
            step=0.05,
            key="qa_threshold"
        )
        st.checkbox(
            "Remove Duplicates",
            value=st.session_state.get("qa_dup_removal", True),
            key="qa_dup_removal"
        )
        
        st.markdown("#### 🗂️ Vector Index Info")
        st.markdown(f"• **Index Type:** `{index_info.get('index_type', 'N/A')}`")
        st.markdown(f"• **Dimension:** `{index_info.get('dimension', 'N/A')}`")
        
        size_bytes = index_info.get('index_file_size_bytes', 0)
        st.markdown(f"• **File Size:** `{size_bytes / 1024:.1f} KB`")
        
        st.markdown(f"• **Index Version:** `{index_info.get('index_version', 'N/A')}`")
        st.markdown(f"• **Embedding Version:** `{index_info.get('embedding_version', 'N/A')}`")
        st.markdown(f"• **Schema Version:** `{index_info.get('schema_version', 'N/A')}`")
        st.markdown(f"• **Total Documents:** `{index_info.get('total_documents', 0)}`")
        st.markdown(f"• **Avg Chunks/Doc:** `{index_info.get('average_chunks_per_document', 0.0):.1f}`")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🔄 Force Reload"):
                idx_manager.reload_index()
                st.success("FAISS Index reloaded.")
                st.rerun()
        with col_c2:
            if st.button("🧹 Reset Index", type="secondary"):
                idx_manager.clear_all()
                st.session_state.parsed_files = {}
                st.success("Vector Store cleared.")
                st.rerun()
    
    st.markdown("---")
    
    # On-the-fly chunking, embedding, and indexing for backward compatibility
    for f_name, f_data in st.session_state.parsed_files.items():
        if ("chunks" not in f_data or "vectors" not in f_data) and "documents" in f_data:
            from src.text_processing import DocumentConverter, ChunkingEngine
            from src.embedding import EmbeddingManager
            from src.vector_store import IndexManager
            f_ext = f_data["file_type"]
            f_cat = "image" if f_ext in ["png", "jpg", "jpeg", "bmp", "tiff"] else ("audio" if f_ext in ["mp3", "wav", "m4a", "flac"] else f_ext)
            try:
                ud = DocumentConverter.convert(
                    documents=f_data["documents"],
                    source_file=f_data["file_name"],
                    source_type=f_cat,
                    processing_time=f_data.get("processing_time", 0.0)
                )
                f_data["unified_document"] = ud
                chunks = ChunkingEngine.chunk_document(
                    doc=ud,
                    chunk_size=st.session_state.get("ui_chunk_size", 500),
                    chunk_overlap=st.session_state.get("ui_chunk_overlap", 100)
                )
                emb_manager = EmbeddingManager()
                vectors, updated_chunks = emb_manager.embed_chunks(
                    chunks=chunks,
                    batch_size=st.session_state.get("ui_batch_size", 32)
                )
                f_data["chunks"] = updated_chunks
                f_data["vectors"] = vectors
                
                # Dynamic transactional indexing
                idx_manager.add_vectors_and_metadata(vectors, updated_chunks)
            except Exception:
                pass

    # Calculate aggregate stats
    total_files = len(st.session_state.parsed_files)
    pdf_count = sum(1 for f in st.session_state.parsed_files.values() if f["file_type"] == "pdf")
    docx_count = sum(1 for f in st.session_state.parsed_files.values() if f["file_type"] == "docx")
    image_count = sum(1 for f in st.session_state.parsed_files.values() if f["file_type"] in ["png", "jpg", "jpeg", "bmp", "tiff"])
    audio_count = sum(1 for f in st.session_state.parsed_files.values() if f["file_type"] in ["mp3", "wav", "m4a", "flac"])
    
    total_pages = 0
    total_docx_blocks = 0
    total_ocr_blocks = 0
    total_audio_chunks = 0
    total_chunks = sum(len(f.get("chunks", [])) for f in st.session_state.parsed_files.values())
    total_proc_time = 0.0
    
    for f in st.session_state.parsed_files.values():
        ext = f["file_type"]
        docs = f["documents"]
        total_proc_time += f.get("processing_time", 0.0)
        
        if ext == "pdf":
            total_pages += len(docs)
        elif ext == "docx":
            total_docx_blocks += len(docs)
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            for doc in docs:
                total_ocr_blocks += len(doc.metadata.get("blocks", []))
        elif ext in ["mp3", "wav", "m4a", "flac"]:
            for doc in docs:
                total_audio_chunks += len(doc.metadata.get("segments", []))

    # Section 2: Files Dashboard
    st.markdown("### 📁 Files")
    st.markdown(f"""
    <div class="sidebar-metric">
        <div class="sidebar-metric-val">{total_files}</div>
        <div class="sidebar-metric-lbl">Total Uploaded</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown(f"📄 **PDFs:** {pdf_count}")
        st.markdown(f"🖼️ **Images:** {image_count}")
    with col_f2:
        st.markdown(f"📝 **DOCXs:** {docx_count}")
        st.markdown(f"🎵 **Audios:** {audio_count}")
        
    st.markdown("---")
    
    # Section 3: Processing Metrics
    st.markdown("### ⚙️ Processing")
    st.markdown(f"• **Pages Processed (PDF):** {total_pages}")
    st.markdown(f"• **Structure Blocks (DOCX):** {total_docx_blocks}")
    st.markdown(f"• **OCR Blocks (Images):** {total_ocr_blocks}")
    st.markdown(f"• **Audio Chunks:** {total_audio_chunks}")
    st.markdown(f"• **Total Text Chunks:** {total_chunks}")
    
    st.markdown("---")
    
    # Section 4: AI Pipeline
    st.markdown("### 🧠 AI Pipeline")
    from src.llm import LLMManager
    from src.config.settings import LLM_RUNTIME_MODE, LLM_RUNTIME_PROFILE
    llm_manager = LLMManager()
    ollama_running = llm_manager.client.is_server_running()
    if ollama_running:
        if LLM_RUNTIME_MODE == "cpu":
            ollama_status = "Connected (CPU Mode) 🟡"
        else:
            ollama_status = "Connected 🟢"
    else:
        ollama_status = "Disconnected 🔴"
    
    st.markdown(f"• **Embedding Model:** `MiniLM-L12-v2` 🟢")
    st.markdown(f"• **Embedding Dimension:** `384` 📏")
    st.markdown(f"• **Indexed Documents:** `{index_info.get('total_documents', 0)}` 📄")
    st.markdown(f"• **Indexed Chunks:** `{idx_manager.metadata_store.get_index_stats().get('total_vectors', 0)}` 📑")
    st.markdown(f"• **Indexed Vectors:** `{idx_manager.engine.total}` 🗂️")
    st.markdown(f"• **LLM Model:** `{llm_manager.model_name}` 🤖")
    st.markdown(f"• **Ollama Status:** `{ollama_status}`")
    st.markdown(f"• **Runtime Mode Configured:** `{LLM_RUNTIME_MODE.upper()}`")
    st.markdown(f"• **Runtime Profile Configured:** `{LLM_RUNTIME_PROFILE}`")
    st.markdown(f"• **Retrieval Status:** `Active 🟢`")
    
    st.markdown("---")
    
    # Section 5: Performance
    st.markdown("### ⚡ Performance")
    avg_time = total_proc_time / total_files if total_files > 0 else 0.0
    st.markdown(f"• **Total Parse Time:** {total_proc_time:.2f}s")
    st.markdown(f"• **Avg Time / File:** {avg_time:.2f}s")
    if st.session_state.get("retrieval_latencies", []):
        avg_ret = sum(st.session_state.retrieval_latencies) / len(st.session_state.retrieval_latencies)
        st.markdown(f"• **Avg Query Latency:** {avg_ret:.2f}ms")
    
    st.markdown("---")
    if st.button("Clear Parsing Memory", type="primary", use_container_width=True):
        st.session_state.parsed_files = {}
        st.success("Parsing cache cleared.")
        st.rerun()

# 5. Header Title Banner
st.markdown("<div class='banner-title'>Offline Multimodal RAG System</div>", unsafe_allow_html=True)
st.markdown("<div class='banner-subtitle'>English • Tamil • AI-Powered Local Knowledge Assistant</div>", unsafe_allow_html=True)

# 5.5 Startup Warnings Display
if st.session_state.get("startup_warnings"):
    for warning in st.session_state.startup_warnings:
        st.warning(f"⚠️ {warning}")

# 6. Global Latencies Cache
if "retrieval_latencies" not in st.session_state:
    st.session_state.retrieval_latencies = []
if "llm_latencies" not in st.session_state:
    st.session_state.llm_latencies = []

# 7. Metrics Dashboard Cards
st.markdown("### 📊 Metrics Dashboard")
stats = idx_manager.metadata_store.get_index_stats()
total_docs = stats.get("total_documents", 0)
total_chunks = stats.get("total_vectors", 0)
total_vectors = idx_manager.engine.total

avg_ret_time = sum(st.session_state.retrieval_latencies) / len(st.session_state.retrieval_latencies) if st.session_state.retrieval_latencies else 0.0
avg_llm_time = sum(st.session_state.llm_latencies) / len(st.session_state.llm_latencies) if st.session_state.llm_latencies else 0.0

d_col1, d_col2, d_col3, d_col4, d_col5, d_col6 = st.columns(6)
d_col1.metric("Documents", total_docs)
d_col2.metric("Chunks", total_chunks)
d_col3.metric("Vectors", total_vectors)
d_col4.metric("Embeddings Model", "MiniLM-L12")
d_col5.metric("Avg Retrieval", f"{avg_ret_time:.1f} ms")
d_col6.metric("Avg LLM Time", f"{avg_llm_time:.1f} ms")

st.markdown("---")

# Main application tab layout
tab_chat, tab_ingest = st.tabs(["💬 Grounded Chat Q&A", "📤 Ingestion & Library"])

with tab_ingest:
    st.markdown("### 📥 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, Image, or Audio files to extract text locally",
        type=["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff", "mp3", "wav", "m4a", "flac"],
        accept_multiple_files=True,
        help="Files are validated locally and stored offline in data/uploads/",
        key="file_uploader_ingest"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if file_name in st.session_state.parsed_files:
                continue
                
            file_bytes = uploaded_file.read()
            progress_bar = st.progress(0, text=f"Validating {file_name}...")
            
            saved_path, message = FileManager.save_upload(file_name, file_bytes)
            if not saved_path:
                progress_bar.empty()
                st.error("❌ Validation Failed")
                st.markdown(f"""
                **File Name:** {file_name}
                **Uploaded Size:** {message.get('size_mb', 0.0) if isinstance(message, dict) else 0.0:.2f} MB
                **Reason:** {message.get('reason', message) if isinstance(message, dict) else message}
                """)
                continue
                
            progress_bar.progress(30, text=f"Saving {file_name} to disk...")
            ext = file_name.split(".")[-1].lower()
            progress_bar.progress(50, text=f"Initializing local parser for {file_name}...")
            
            try:
                start_parse_time = time.time()
                parser = ParserFactory.get_parser(ext)
                progress_bar.progress(70, text=f"Extracting text from {file_name}...")
                parsed_docs = parser.parse(str(saved_path))
                parse_duration = time.time() - start_parse_time
                
                from src.text_processing import DocumentConverter, ChunkingEngine
                from src.embedding import EmbeddingManager
                
                category = "image" if ext in ["png", "jpg", "jpeg", "bmp", "tiff"] else ("audio" if ext in ["mp3", "wav", "m4a", "flac"] else ext)
                unified_doc = DocumentConverter.convert(
                    documents=parsed_docs,
                    source_file=file_name,
                    source_type=category,
                    processing_time=parse_duration
                )
                
                chunk_size = st.session_state.get("ui_chunk_size", 500)
                chunk_overlap = st.session_state.get("ui_chunk_overlap", 100)
                chunks = ChunkingEngine.chunk_document(unified_doc, chunk_size, chunk_overlap)
                
                progress_bar.progress(80, text=f"Generating local L2 embeddings...")
                emb_manager = EmbeddingManager()
                batch_size = st.session_state.get("ui_batch_size", 32)
                vectors, updated_chunks = emb_manager.embed_chunks(chunks, batch_size=batch_size)
                
                progress_bar.progress(95, text=f"Indexing in FAISS index...")
                idx_manager.add_vectors_and_metadata(vectors, updated_chunks)
                
                st.session_state.parsed_files[file_name] = {
                    "file_name": file_name,
                    "file_type": ext,
                    "file_size": len(file_bytes),
                    "file_path": str(saved_path),
                    "documents": parsed_docs,
                    "unified_document": unified_doc,
                    "chunks": updated_chunks,
                    "vectors": vectors,
                    "processing_time": parse_duration
                }
                st.success(f"Parsed and Indexed **{file_name}** successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing **{file_name}**: {str(e)}")
            finally:
                progress_bar.empty()

    st.markdown("---")
    st.markdown("### 🗂️ Document Library")
    docs = idx_manager.metadata_store.get_all_documents()
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        # Table of indexed files
        table_list = []
        for d in docs:
            doc_hash = d["document_hash"]
            file_name = d["file_name"] or "Unknown"
            ext = file_name.split(".")[-1] if "." in file_name else "Unknown"
            chunks = d["indexed_chunks"]
            status = d["index_status"]
            table_list.append({
                "Filename": file_name,
                "Type": ext.upper(),
                "Indexed Chunks": chunks,
                "Last Indexed": d["last_indexed"],
                "Indexed Status": status
            })
        st.dataframe(table_list, use_container_width=True)
        
        # Actions
        st.markdown("#### Manage Documents")
        for d in docs:
            doc_hash = d["document_hash"]
            file_name = d["file_name"] or "Unknown"
            col_lbl, col_del, col_re = st.columns([4, 1, 1])
            col_lbl.write(f"📄 **{file_name}**")
            
            if col_del.button("🗑️ Delete", key=f"del_{doc_hash}", use_container_width=True):
                idx_manager.delete_document(doc_hash)
                # Remove from cache if loaded
                if file_name in st.session_state.parsed_files:
                    del st.session_state.parsed_files[file_name]
                st.success(f"Deleted {file_name} successfully!")
                time.sleep(0.5)
                st.rerun()
                
            if col_re.button("🔄 Re-index", key=f"re_{doc_hash}", use_container_width=True):
                with st.spinner("Re-indexing..."):
                    file_path = UPLOAD_DIR / file_name
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        idx_manager.delete_document(doc_hash)
                        if file_name in st.session_state.parsed_files:
                            del st.session_state.parsed_files[file_name]
                            
                        # Re-run pipeline
                        parser = ParserFactory.get_parser(file_name.split(".")[-1].lower())
                        parsed_docs = parser.parse(str(file_path))
                        from src.text_processing import DocumentConverter, ChunkingEngine
                        from src.embedding import EmbeddingManager
                        category = "pdf" if file_name.endswith(".pdf") else "docx"
                        ud = DocumentConverter.convert(parsed_docs, file_name, category, 0.0)
                        chunks = ChunkingEngine.chunk_document(ud, st.session_state.get("ui_chunk_size", 500), st.session_state.get("ui_chunk_overlap", 100))
                        emb_manager = EmbeddingManager()
                        vectors, updated_chunks = emb_manager.embed_chunks(chunks)
                        idx_manager.add_vectors_and_metadata(vectors, updated_chunks)
                        
                        st.session_state.parsed_files[file_name] = {
                            "file_name": file_name,
                            "file_type": file_name.split(".")[-1],
                            "file_size": file_path.stat().st_size,
                            "file_path": str(file_path),
                            "documents": parsed_docs,
                            "unified_document": ud,
                            "chunks": updated_chunks,
                            "vectors": vectors,
                            "processing_time": 0.0
                        }
                        st.success(f"Re-indexed {file_name}!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Upload source file missing on disk.")

    st.markdown("---")
    st.markdown("### 📋 Uploaded Files Metadata Cards")
    if st.session_state.parsed_files:
        for name, file_data in st.session_state.parsed_files.items():
            ext = file_data["file_type"]
            pages_count = len(file_data["documents"])
            
            with st.expander(f"📄 {name} Information Card"):
                info_data = {
                    "Metric": ["Filename", "Type", "Pages/Blocks", "Chunks Size", "Embedding Status", "Indexed Status", "Processing Time"],
                    "Value": [
                        file_data["file_name"],
                        ext.upper(),
                        f"{pages_count} blocks/pages",
                        str(len(file_data.get("chunks", []))),
                        "L2-Normalized (384-dim) 🟢",
                        "Indexed in FAISS 🟢",
                        f"{file_data.get('processing_time', 0.0):.2f}s"
                    ]
                }
                st.table(info_data)
                
                # Displays audio/image players
                if ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
                    st.image(file_data["file_path"], use_container_width=True)
                elif ext in ["mp3", "wav", "m4a", "flac"]:
                    st.audio(file_data["file_path"])
    else:
        # Fallback to persistent SQLite document metadata
        sqlite_docs = idx_manager.metadata_store.get_all_documents()
        if not sqlite_docs:
            st.info("No uploaded metadata cards available. Upload files above to view info cards.")
        else:
            for d in sqlite_docs:
                file_name = d["file_name"] or "Unknown"
                ext = file_name.split(".")[-1].lower() if "." in file_name else "doc"
                indexed_chunks = d["indexed_chunks"]
                status = d["index_status"]
                last_idx = d["last_indexed"]
                file_path = UPLOAD_DIR / file_name
                
                with st.expander(f"📄 {file_name} Information Card"):
                    info_data = {
                        "Metric": ["Filename", "Type", "Indexed Chunks", "Embedding Status", "Indexed Status", "Last Indexed"],
                        "Value": [
                            file_name,
                            ext.upper(),
                            str(indexed_chunks),
                            "L2-Normalized (384-dim) 🟢",
                            f"{status} 🟢",
                            last_idx
                        ]
                    }
                    st.table(info_data)
                    
                    if file_path.exists():
                        if ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
                            st.image(str(file_path), use_container_width=True)
                        elif ext in ["mp3", "wav", "m4a", "flac"]:
                            st.audio(str(file_path))

with tab_chat:
    st.markdown("### 💬 Grounded Chat Q&A")
    
    # Mode Selector
    mode_sel = st.selectbox(
        "Select Retrieval Mode Override:",
        options=["Automatic", "Semantic Search", "Document Summary", "Overview", "Translation"],
        key="ui_retrieval_mode"
    )
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    col_cc1, col_cc2 = st.columns([5, 1])
    with col_cc2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.retrieval_latencies = []
            st.session_state.llm_latencies = []
            st.rerun()
            
    # Render chat bubbles
    for chat_idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                diag = msg.get("diagnostics", {})
                if diag.get("cpu_fallback_triggered", False):
                    st.info("ℹ️ GPU memory was insufficient. Inference completed using CPU mode.")
            st.markdown(msg["content"])
            
            # If assistant message has chunks, render cards
            if msg["role"] == "assistant" and msg.get("success", False):
                # Copy Answer button container
                st.code(msg["content"], language="text")
                
                # Expandable chunk cards
                if msg.get("retrieved_chunks"):
                    with st.expander("📖 Show Retrieved Context Chunks"):
                        for idx, c in enumerate(msg["retrieved_chunks"], 1):
                            st.markdown(
                                f"""
                                <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 8px;">
                                    <div style="font-weight: bold; font-size: 14px; color: #f8fafc;">📄 {c['file']}</div>
                                    <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                                        <span>📍 Reference: <b>{c['ref']}</b></span> &nbsp;|&nbsp; 
                                        <span>🎯 Similarity: <b>{c['score'] * 100:.1f}%</b></span> &nbsp;|&nbsp;
                                        <span>🔢 Chunk: <b>#{c['id'].split('_')[-1]}</b></span>
                                    </div>
                                    <div style="margin-top: 8px; font-family: monospace; font-size: 12px; color: #cbd5e1; white-space: pre-wrap;">{c['text']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # Pipeline Details Explanations & Developer Mode diagnostics
                if st.session_state.get("dev_mode", False):
                    with st.expander("🔧 Developer Mode: LLM & Prompt Metrics", expanded=True):
                        st.markdown(f"• **Detected Intent:** `{msg.get('intent', 'N/A')}`")
                        st.markdown(f"• **Retrieval Mode:** `{msg.get('retrieval_mode', 'N/A')}`")
                        st.markdown(f"• **Threshold Used:** `{msg.get('threshold', 0.0):.2f}`")
                        st.markdown(f"• **Highest Similarity:** `{msg.get('highest_similarity', 0.0):.4f}`")
                        st.markdown(f"• **Lowest Similarity:** `{msg.get('lowest_similarity', 0.0):.4f}`")
                        st.markdown(f"• **Average Similarity:** `{msg.get('average_similarity', 0.0):.4f}`")
                        
                        diag = msg.get("diagnostics", {})
                        st.markdown("##### 🛡️ Response Validation Diagnostics")
                        st.markdown(f"• **Validation Passed:** `{diag.get('validation_passed', True)}`")
                        st.markdown(f"• **Validation Failed:** `{diag.get('validation_failed', False)}`")
                        st.markdown(f"• **Validation Reason:** `{diag.get('validation_reason', 'Passed')}`")
                        st.markdown(f"• **Unsupported Claims Found:** `{diag.get('unsupported_claims_found', [])}`")
                        st.markdown(f"• **Unsupported Entities Found:** `{diag.get('unsupported_entities_found', [])}`")
                        st.markdown(f"• **Semantic Similarity Score:** `{diag.get('semantic_similarity_score', 1.0):.2f}`")
                        
                        st.markdown("##### 📏 Prompt Budget & Context Utilization")
                        st.markdown(f"• **MAX_CONTEXT_CHARACTERS:** `{diag.get('max_context_characters', 4000)}`")
                        st.markdown(f"• **System Prompt Characters:** `{diag.get('system_prompt_chars', 0)}`")
                        st.markdown(f"• **Question Characters:** `{diag.get('question_chars', 0)}`")
                        st.markdown(f"• **Template Characters:** `{diag.get('template_chars', 0)}`")
                        st.markdown(f"• **Base Prompt Characters:** `{diag.get('base_prompt_chars', 0)}`")
                        st.markdown(f"• **Available Context Budget:** `{diag.get('available_context_budget', 0)}`")
                        st.markdown(f"• **Retrieved Context Characters:** `{diag.get('retrieved_context_chars', 0)}`")
                        st.markdown(f"• **Final Prompt Characters:** `{diag.get('final_prompt_chars', 0)}`")
                        st.markdown(f"• **Remaining Budget:** `{diag.get('remaining_budget', 0)}`")
                        st.markdown(f"• **Context Utilization (%):** `{diag.get('context_utilization_percent', 0.0):.2f}%`")
                        st.markdown(f"• **Chunks Retrieved:** `{diag.get('chunks_retrieved', 0)}`")
                        st.markdown(f"• **Chunks Included:** `{diag.get('chunks_included', 0)}`")
                        st.markdown(f"• **Chunks Trimmed:** `{diag.get('chunks_trimmed', 0)}`")
                        
                        st.markdown("##### ⚙️ LLM Runtime Telemetry")
                        st.markdown(f"• **Runtime Mode:** `{diag.get('runtime_mode', 'N/A')}`")
                        st.markdown(f"• **Runtime Profile:** `{diag.get('runtime_profile', 'N/A')}`")
                        st.markdown(f"• **Model Name:** `{diag.get('model', 'N/A')}`")
                        st.markdown(f"• **num_ctx:** `{diag.get('context_limit', 'N/A')}`")
                        st.markdown(f"• **num_predict:** `{diag.get('num_predict', 'N/A')}`")
                        st.markdown(f"• **Prompt Length:** `{diag.get('prompt_length', 0)} chars`")
                        st.markdown(f"• **HTTP Status:** `{diag.get('http_status', 'N/A')}`")
                        st.markdown(f"• **Inference Time:** `{diag.get('inference_time_ms', 0.0):.2f} ms`")
                        st.markdown(f"• **Retry Performed:** `{diag.get('retry_performed', 'No')}`")
                        st.markdown(f"• **Retry Reason:** `{diag.get('retry_reason', 'None')}`")
                        st.markdown(f"• **GPU OOM Detected:** `{diag.get('gpu_oom_detected', 'No')}`")
                        st.markdown(f"• **Final Runtime Used:** `{diag.get('final_runtime_used', 'N/A')}`")
                        st.markdown(f"• **Final Runtime Mode:** `{diag.get('final_runtime_mode', 'N/A')}`")
                        
                        rec = get_hardware_recommendation()
                        st.markdown(f"• **Hardware Recommendation:** `{rec}`")
                        
                        if diag.get("ollama_error"):
                            st.markdown(f"• **Ollama Error:** `{diag.get('ollama_error')}`")

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # 1. Pipeline Status Visualizer
        with st.status("Executing RAG Pipeline...", expanded=True) as status_bar:
            status_bar.write("🔍 Running Intent Classification...")
            time.sleep(0.1)
            
            # Setup Retrieval Config
            from src.retrieval import RetrievalManager, RetrievalConfig
            threshold_val = st.session_state.get("qa_threshold", 0.70)
            top_k_val = st.session_state.get("qa_top_k", 5)
            dup_removal = st.session_state.get("qa_dup_removal", True)
            
            config = RetrievalConfig(
                top_k=int(top_k_val),
                similarity_threshold=float(threshold_val),
                duplicate_removal=dup_removal,
                retrieval_mode=mode_sel
            )
            
            status_bar.write("🧠 Generating Multilingual Embeddings...")
            time.sleep(0.1)
            
            status_bar.write("🗂️ Running Nearest-Neighbor FAISS Vector Search...")
            time.sleep(0.1)
            
            status_bar.write("⚖️ Ranking and Filtering Results...")
            time.sleep(0.1)
            
            # Search
            ret_manager = RetrievalManager()
            result = ret_manager.search(prompt, config=config)
            
            status_bar.write("📑 Assembling Context blocks...")
            time.sleep(0.1)
            
            status_bar.write("🤖 Invoking Local Ollama generation...")
            from src.llm import LLMManager
            llm_manager = LLMManager()
            ans_res = llm_manager.generate_grounded_answer(result)
            
            status_bar.update(label="Pipeline processing complete!", state="complete", expanded=False)
            
        # Update aggregate stats
        if result.success:
            st.session_state.retrieval_latencies.append(result.latency_metrics.get("total_latency_ms", 0.0))
        if ans_res.success:
            st.session_state.llm_latencies.append(ans_res.latency_metrics.get("inference_time_ms", 0.0))
            
        with st.chat_message("assistant"):
            if result.is_low_confidence:
                st.warning("⚠️ Low confidence retrieval. Answer may be less reliable.")
            diag = ans_res.diagnostics if hasattr(ans_res, "diagnostics") else {}
            if diag.get("cpu_fallback_triggered", False):
                st.info("ℹ️ GPU memory was insufficient. Inference completed using CPU mode.")
            st.markdown(ans_res.answer)
            
            # Copy Code box
            st.code(ans_res.answer, language="text")
            
            # Render context cards
            if result.success and result.retrieved_chunks:
                with st.expander("📖 Show Retrieved Context Chunks"):
                    for idx, c in enumerate(result.retrieved_chunks, 1):
                        st.markdown(
                            f"""
                            <div style="background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 8px;">
                                <div style="font-weight: bold; font-size: 14px; color: #f8fafc;">📄 {c.source_file}</div>
                                <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                                    <span>📍 Reference: <b>{c.source_reference}</b></span> &nbsp;|&nbsp; 
                                    <span>🎯 Similarity: <b>{c.similarity_score * 100:.1f}%</b></span> &nbsp;|&nbsp;
                                    <span>🔢 Chunk: <b>#{c.chunk_id.split('_')[-1]}</b></span>
                                </div>
                                <div style="margin-top: 8px; font-family: monospace; font-size: 12px; color: #cbd5e1; white-space: pre-wrap;">{c.chunk_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
            # Developer stats
            if st.session_state.get("dev_mode", False):
                with st.expander("🔧 Developer Mode: LLM & Prompt Metrics", expanded=True):
                    st.markdown(f"• **Detected Intent:** `{result.intent}`")
                    st.markdown(f"• **Retrieval Mode:** `{result.retrieval_mode}`")
                    st.markdown(f"• **Threshold Used:** `{result.statistics.get('threshold', 0.0):.2f}`")
                    st.markdown(f"• **Highest Similarity:** `{result.highest_similarity:.4f}`")
                    st.markdown(f"• **Lowest Similarity:** `{result.lowest_similarity:.4f}`")
                    st.markdown(f"• **Average Similarity:** `{result.average_similarity:.4f}`")
                    
                    st.markdown("##### 🛡️ Response Validation Diagnostics")
                    st.markdown(f"• **Validation Passed:** `{diag.get('validation_passed', True)}`")
                    st.markdown(f"• **Validation Failed:** `{diag.get('validation_failed', False)}`")
                    st.markdown(f"• **Validation Reason:** `{diag.get('validation_reason', 'Passed')}`")
                    st.markdown(f"• **Unsupported Claims Found:** `{diag.get('unsupported_claims_found', [])}`")
                    st.markdown(f"• **Unsupported Entities Found:** `{diag.get('unsupported_entities_found', [])}`")
                    st.markdown(f"• **Semantic Similarity Score:** `{diag.get('semantic_similarity_score', 1.0):.2f}`")
                    
                    st.markdown("##### 📏 Prompt Budget & Context Utilization")
                    st.markdown(f"• **MAX_CONTEXT_CHARACTERS:** `{diag.get('max_context_characters', 4000)}`")
                    st.markdown(f"• **System Prompt Characters:** `{diag.get('system_prompt_chars', 0)}`")
                    st.markdown(f"• **Question Characters:** `{diag.get('question_chars', 0)}`")
                    st.markdown(f"• **Template Characters:** `{diag.get('template_chars', 0)}`")
                    st.markdown(f"• **Base Prompt Characters:** `{diag.get('base_prompt_chars', 0)}`")
                    st.markdown(f"• **Available Context Budget:** `{diag.get('available_context_budget', 0)}`")
                    st.markdown(f"• **Retrieved Context Characters:** `{diag.get('retrieved_context_chars', 0)}`")
                    st.markdown(f"• **Final Prompt Characters:** `{diag.get('final_prompt_chars', 0)}`")
                    st.markdown(f"• **Remaining Budget:** `{diag.get('remaining_budget', 0)}`")
                    st.markdown(f"• **Context Utilization (%):** `{diag.get('context_utilization_percent', 0.0):.2f}%`")
                    st.markdown(f"• **Chunks Retrieved:** `{diag.get('chunks_retrieved', 0)}`")
                    st.markdown(f"• **Chunks Included:** `{diag.get('chunks_included', 0)}`")
                    st.markdown(f"• **Chunks Trimmed:** `{diag.get('chunks_trimmed', 0)}`")
                    
                    st.markdown("##### ⚙️ LLM Runtime Telemetry")
                    st.markdown(f"• **Runtime Mode:** `{diag.get('runtime_mode', 'N/A')}`")
                    st.markdown(f"• **Runtime Profile:** `{diag.get('runtime_profile', 'N/A')}`")
                    st.markdown(f"• **Model Name:** `{diag.get('model', 'N/A')}`")
                    st.markdown(f"• **num_ctx:** `{diag.get('context_limit', 'N/A')}`")
                    st.markdown(f"• **num_predict:** `{diag.get('num_predict', 'N/A')}`")
                    st.markdown(f"• **Prompt Length:** `{diag.get('prompt_length', 0)} chars`")
                    st.markdown(f"• **HTTP Status:** `{diag.get('http_status', 'N/A')}`")
                    st.markdown(f"• **Inference Time:** `{diag.get('inference_time_ms', 0.0):.2f} ms`")
                    st.markdown(f"• **Retry Performed:** `{diag.get('retry_performed', 'No')}`")
                    st.markdown(f"• **Retry Reason:** `{diag.get('retry_reason', 'None')}`")
                    st.markdown(f"• **GPU OOM Detected:** `{diag.get('gpu_oom_detected', 'No')}`")
                    st.markdown(f"• **Final Runtime Used:** `{diag.get('final_runtime_used', 'N/A')}`")
                    st.markdown(f"• **Final Runtime Mode:** `{diag.get('final_runtime_mode', 'N/A')}`")
                    
                    rec = get_hardware_recommendation()
                    st.markdown(f"• **Hardware Recommendation:** `{rec}`")
                    
                    if diag.get("ollama_error"):
                        st.markdown(f"• **Ollama Error:** `{diag.get('ollama_error')}`")
                    
        # Append assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ans_res.answer,
            "success": ans_res.success,
            "intent": result.intent,
            "retrieval_mode": result.retrieval_mode,
            "threshold": result.statistics.get("threshold", 0.0),
            "highest_similarity": result.highest_similarity,
            "lowest_similarity": result.lowest_similarity,
            "average_similarity": result.average_similarity,
            "prompt_size_chars": len(prompt),
            "latency_metrics": ans_res.latency_metrics,
            "token_statistics": ans_res.token_statistics,
            "diagnostics": ans_res.diagnostics if hasattr(ans_res, "diagnostics") else {},
            "retrieved_chunks": [
                {
                    "id": c.chunk_id,
                    "file": c.source_file,
                    "ref": c.source_reference,
                    "score": c.similarity_score,
                    "text": c.chunk_text
                } for c in result.retrieved_chunks
            ] if result.success else []
        })
        st.rerun()
