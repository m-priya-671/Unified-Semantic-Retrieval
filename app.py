import streamlit as st
import os
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

# 3. Initialize Session State
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
    
    st.markdown("---")
    
    # On-the-fly chunking calculations for backward compatibility
    for f_name, f_data in st.session_state.parsed_files.items():
        if "chunks" not in f_data and "documents" in f_data:
            from src.text_processing import DocumentConverter, ChunkingEngine
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
                f_data["chunks"] = ChunkingEngine.chunk_document(
                    doc=ud,
                    chunk_size=st.session_state.get("ui_chunk_size", 500),
                    chunk_overlap=st.session_state.get("ui_chunk_overlap", 100)
                )
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
    st.markdown("• **Embeddings Status:** *N/A (Coming in M5)*")
    st.markdown("• **Vector Index Status:** *N/A (Coming in M5/6)*")
    st.markdown("• **LLM Model Configured:** *N/A (Coming in M8)*")
    
    st.markdown("---")
    
    # Section 5: Performance
    st.markdown("### ⚡ Performance")
    avg_time = total_proc_time / total_files if total_files > 0 else 0.0
    st.markdown(f"• **Total Parse Time:** {total_proc_time:.2f}s")
    st.markdown(f"• **Avg Time / File:** {avg_time:.2f}s")
    st.markdown("• **Query Latency:** *N/A (Coming in M7)*")
    
    st.markdown("---")
    if st.button("Clear Parsing Memory", type="primary", use_container_width=True):
        st.session_state.parsed_files = {}
        st.success("Parsing cache cleared.")
        st.rerun()

# 5. Header Title Banner
st.markdown("<div class='banner-title'>Unified Semantic Retrieval</div>", unsafe_allow_html=True)
st.markdown("<div class='banner-subtitle'>Milestone 2: Offline Document Parser & OCR Engine</div>", unsafe_allow_html=True)

# 6. Upload Component
st.markdown("### 📥 Document Ingestion")
uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, Image, or Audio files to extract text locally",
    type=["pdf", "docx", "png", "jpg", "jpeg", "bmp", "tiff", "mp3", "wav", "m4a", "flac"],
    accept_multiple_files=True,
    help="Files are validated locally and stored offline in data/uploads/"
)

# Process uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        
        # Avoid processing duplicate files in the same run
        if file_name in st.session_state.parsed_files:
            continue
            
        # File manager validation and saving
        file_bytes = uploaded_file.read()
        
        # Setup streamlit progress
        progress_bar = st.progress(0, text=f"Validating {file_name}...")
        
        saved_path, message = FileManager.save_upload(file_name, file_bytes)
        
        if not saved_path:
            progress_bar.empty()
            if isinstance(message, dict):
                st.error("❌ Validation Failed")
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 18px; margin-bottom: 20px;">
                    <h4 style="color: #f87171; margin-top: 0px; margin-bottom: 12px; font-family: 'Outfit', sans-serif;">⚠️ File Validation Error Report</h4>
                    <table style="width: 100%; border-collapse: collapse; font-family: sans-serif;">
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; width: 140px; font-size: 0.9rem;">File Name</td><td style="padding: 5px 0; color: #f8fafc; font-size: 0.9rem;">{message['file_name']}</td></tr>
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; font-size: 0.9rem;">Uploaded Size</td><td style="padding: 5px 0; color: #f8fafc; font-size: 0.9rem;">{message['size_mb']:.2f} MB</td></tr>
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; font-size: 0.9rem;">Allowed Size</td><td style="padding: 5px 0; color: #f8fafc; font-size: 0.9rem;">{message['limit_mb']:.1f} MB</td></tr>
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; font-size: 0.9rem;">Status</td><td style="padding: 5px 0; color: #f87171; font-weight: bold; font-size: 0.9rem;">Validation Failed</td></tr>
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; font-size: 0.9rem;">Reason</td><td style="padding: 5px 0; color: #e2e8f0; font-size: 0.9rem;">{message['reason']}</td></tr>
                        <tr><td style="padding: 5px 0; color: #f87171; font-weight: 600; font-size: 0.9rem;">Suggestion</td><td style="padding: 5px 0; color: #38bdf8; font-style: italic; font-size: 0.9rem;">{message['suggestion']}</td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Failed to process **{file_name}**: {message}")
            continue
            
        progress_bar.progress(30, text=f"Saving {file_name} to disk...")
        
        # Get extension
        ext = file_name.split(".")[-1].lower()
        
        progress_bar.progress(50, text=f"Initializing local parser for {file_name}...")
        
        try:
            start_parse_time = time.time()
            parser = ParserFactory.get_parser(ext)
            progress_bar.progress(70, text=f"Extracting text from {file_name}...")
            
            parsed_docs = parser.parse(str(saved_path))
            parse_duration = time.time() - start_parse_time
            
            # Store in session state
            # Convert and Chunk
            from src.text_processing import DocumentConverter, ChunkingEngine
            category = "image" if ext in ["png", "jpg", "jpeg", "bmp", "tiff"] else ("audio" if ext in ["mp3", "wav", "m4a", "flac"] else ext)
            unified_doc = DocumentConverter.convert(
                documents=parsed_docs,
                source_file=file_name,
                source_type=category,
                processing_time=parse_duration
            )
            
            chunk_size = st.session_state.get("ui_chunk_size", 500)
            chunk_overlap = st.session_state.get("ui_chunk_overlap", 100)
            
            chunks = ChunkingEngine.chunk_document(
                doc=unified_doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            st.session_state.parsed_files[file_name] = {
                "file_name": file_name,
                "file_type": ext,
                "file_size": len(file_bytes),
                "file_path": str(saved_path),
                "documents": parsed_docs,
                "unified_document": unified_doc,
                "chunks": chunks,
                "processing_time": parse_duration
            }
            
            progress_bar.progress(100, text=f"Successfully extracted {file_name}!")
            st.success(f"Parsed **{file_name}** ({len(file_bytes)/1024:.1f} KB) - Extracted {len(parsed_docs)} structures.")
            
        except Exception as e:
            st.error(f"Error parsing **{file_name}**: {str(e)}")
            logger.error(f"Parsing execution failed for file {file_name}: {str(e)}")
            
        # Remove progress bar cleanly after delay
        progress_bar.empty()

# 7. Parsed Content Visualization
st.markdown("### 📋 Ingested Documents List")

if total_files == 0:
    st.info("No documents uploaded yet. Upload PDF, DOCX, or Image files above to parse text.")
else:
    for name, file_data in st.session_state.parsed_files.items():
        # Extracted card structure
        ext = file_data["file_type"]
        if ext == "pdf":
            badge_cls = "badge-pdf"
        elif ext == "docx":
            badge_cls = "badge-docx"
        elif ext in ["mp3", "wav", "m4a", "flac"]:
            badge_cls = "badge-audio"
        else:
            badge_cls = "badge-image"
            
        # Collect dynamic card details
        file_size_mb = file_data["file_size"] / (1024 * 1024)
        docs_list = file_data["documents"]
        first_doc_meta = docs_list[0].metadata if docs_list else {}
        
        parser_used = "Unknown"
        ocr_engine = "N/A"
        pages_count = "N/A"
        blocks_count = "N/A"
        resolution = "N/A"
        audio_duration = "N/A"
        audio_channels = "N/A"
        audio_sr = "N/A"
        languages_list = []
        upload_time = first_doc_meta.get("extracted_at", "Just now")
        
        if ext == "pdf":
            parser_used = "PDFParser"
            pages_count = f"{first_doc_meta.get('total_pages', len(docs_list))}"
            pipeline_steps = ["Upload", "Validation", "Text Extraction", "Metadata Generated", "Ready for Chunking"]
        elif ext == "docx":
            parser_used = "DocxParser"
            blocks_count = f"{len(docs_list)}"
            pipeline_steps = ["Upload", "Validation", "Structure Parsing", "Metadata Generated", "Ready for Chunking"]
        elif ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            parser_used = "ImageProcessor"
            ocr_engine = first_doc_meta.get("ocr_engine", "Unknown")
            dims = first_doc_meta.get("dimensions", {})
            if dims:
                resolution = f"{dims.get('width', 0)} x {dims.get('height', 0)}"
            pipeline_steps = ["Upload", "Validation", "Image Preprocessing", "OCR Extraction", "Metadata Generated", "Ready for Chunking"]
        elif ext in ["mp3", "wav", "m4a", "flac"]:
            parser_used = "AudioProcessor"
            audio_duration = f"{first_doc_meta.get('duration_sec', 0.0):.2f}s"
            audio_channels = "Mono" if first_doc_meta.get("channels", 1) == 1 else "Stereo"
            audio_sr = f"{first_doc_meta.get('sample_rate', 0)} Hz"
            languages_list = first_doc_meta.get("languages", [])
            pipeline_steps = ["Upload", "Validation", "Audio Preprocessing", "Speech Recognition", "Metadata Generated", "Ready for Chunking"]
        else:
            pipeline_steps = ["Upload", "Validation", "Processing", "Metadata Generated", "Ready for Chunking"]

        steps_html = " ➔ ".join(f"<span style='background: rgba(99, 102, 241, 0.15); color: #c084fc; padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; border: 1px solid rgba(168, 85, 247, 0.3); font-weight: 600;'>{step}</span>" for step in pipeline_steps)

        info_table_rows = f"""
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; width: 180px; font-size: 0.9rem;">File Name</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{file_data['file_name']}</td></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">File Type</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem; text-transform: uppercase;">{ext}</td></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">File Size</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{file_size_mb:.2f} MB ({file_data['file_size']/1024:.1f} KB)</td></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Upload Timestamp</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{upload_time}</td></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Processing Status</td><td style="padding: 6px 0; color: #10b981; font-weight: bold; font-size: 0.9rem;">Success 🟢</td></tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Parser Used</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{parser_used}</td></tr>
        """
        
        if ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            info_table_rows += f"""
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">OCR Engine Used</td><td style="padding: 6px 0; color: #34d399; font-weight: bold; font-size: 0.9rem;">{ocr_engine}</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Image Resolution</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{resolution}</td></tr>
            """
        elif ext == "pdf":
            info_table_rows += f"<tr style='border-bottom: 1px solid rgba(255,255,255,0.03);'><td style='padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;'>Number of Pages</td><td style='padding: 6px 0; color: #f8fafc; font-size: 0.9rem;'>{pages_count}</td></tr>"
        elif ext == "docx":
            info_table_rows += f"<tr style='border-bottom: 1px solid rgba(255,255,255,0.03);'><td style='padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;'>Number of Blocks</td><td style='padding: 6px 0; color: #f8fafc; font-size: 0.9rem;'>{blocks_count}</td></tr>"
        elif ext in ["mp3", "wav", "m4a", "flac"]:
            lang_str = ", ".join(f"{l.get('language','Unknown').upper()} ({l.get('probability',0.0)*100:.1f}%)" for l in languages_list)
            info_table_rows += f"""
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Detected Language</td><td style="padding: 6px 0; color: #34d399; font-weight: bold; font-size: 0.9rem;">{lang_str}</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Audio Duration</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{audio_duration}</td></tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);"><td style="padding: 6px 0; font-weight: 600; color: #94a3b8; font-size: 0.9rem;">Channels / Sample Rate</td><td style="padding: 6px 0; color: #f8fafc; font-size: 0.9rem;">{audio_channels} / {audio_sr}</td></tr>
            """

        st.markdown(f"""
        <div class="glass-card" style="padding: 22px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; font-family: 'Outfit', sans-serif;">📋 File Information Card</span>
                <span class="badge {badge_cls}">{ext}</span>
            </div>
            
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                {info_table_rows}
            </table>
            
            <div style="border-top: 1px solid rgba(255,255,255,0.08); padding-top: 15px;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Processing Pipeline</div>
                <div style="line-height: 2.2; margin-left: 5px;">
                    {steps_html}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display preview for images
        if ext in ["png", "jpg", "jpeg", "bmp", "tiff"]:
            st.image(file_data["file_path"], caption=file_data["file_name"], use_container_width=True)
            
            # Display image OCR metrics
            doc = file_data["documents"][0]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("OCR Engine", doc.metadata.get("ocr_engine", "Unknown"))
            with col_b:
                conf = doc.metadata.get("confidence", 0.0)
                st.metric("Avg Confidence", f"{conf * 100:.1f}%")
            with col_c:
                dur = doc.metadata.get("processing_duration_sec", 0.0)
                st.metric("Processing Time", f"{dur:.2f}s")
                
        # Display player for audios
        if ext in ["mp3", "wav", "m4a", "flac"]:
            st.audio(file_data["file_path"])
            
            # Display audio transcription metrics
            doc = file_data["documents"][0]
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("STT Engine", doc.metadata.get("transcription_engine", "Unknown"))
            with col_b:
                langs = doc.metadata.get("languages", [])
                conf_val = langs[0].get("probability", 0.0) if langs else 0.0
                st.metric("Detection Conf.", f"{conf_val * 100:.1f}%")
            with col_c:
                dur = doc.metadata.get("processing_duration_sec", 0.0)
                st.metric("Transcription Time", f"{dur:.2f}s")
        
        # Display structures under this card
        docs_list = file_data["documents"]
        
        # Group page/block view under tabs or expander
        expander_label = f"Show Extracted Content ({len(docs_list)} Blocks/Pages)"
        with st.expander(expander_label, expanded=False):
            # Display pages/blocks cleanly
            for doc in docs_list:
                meta = doc.metadata
                if ext == "pdf":
                    header = f"Page {meta['page_number']} / {meta['total_pages']}"
                elif ext == "docx":
                    header = f"Block {meta['block_index']} ({meta['block_type'].capitalize()})"
                elif ext in ["mp3", "wav", "m4a", "flac"]:
                    header = "Speech Transcript"
                else:
                    header = "OCR Extracted Text"
                    
                st.markdown(f"##### 📍 {header}")
                st.text_area(
                    label=f"Clean Transcript Text ({len(doc.text)} chars)",
                    value=doc.text,
                    height=150,
                    key=f"{name}_{header}",
                    disabled=True
                )
                
                # Show structured audio timeline segment timestamps if available
                if ext in ["mp3", "wav", "m4a", "flac"] and "segments" in meta and meta["segments"]:
                    with st.expander("🕒 Show Transcription Timeline Segments", expanded=False):
                        st.write("Timestamped dialogue timeline:")
                        timeline_md = ""
                        for seg in meta["segments"]:
                            start_min = int(seg['start'] // 60)
                            start_sec = seg['start'] % 60
                            end_min = int(seg['end'] // 60)
                            end_sec = seg['end'] % 60
                            time_lbl = f"`[{start_min:02d}:{start_sec:05.2f} ➔ {end_min:02d}:{end_sec:05.2f}]`"
                            timeline_md += f"{time_lbl} &nbsp; {seg['text']}\n\n"
                        st.markdown(timeline_md, unsafe_allow_html=True)
                
                # Show detailed bounding boxes for images if available
                if ext in ["png", "jpg", "jpeg", "bmp", "tiff"] and "blocks" in meta and meta["blocks"]:
                    with st.expander("🔍 Show Detailed Text Blocks & Bounding Boxes", expanded=False):
                        st.write("Bounding box locations and segment confidences:")
                        st.dataframe(meta["blocks"])
                
                # Show specific metadata dictionary
                st.json(meta, expanded=False)
                st.markdown("---")
                
        # Developer Mode Chunking Details
        if st.session_state.get("dev_mode", False) and "chunks" in file_data:
            chunks = file_data["chunks"]
            with st.expander("🔧 Developer Mode: Chunking Details", expanded=True):
                st.markdown("#### 📊 Chunking Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    st.metric("Total Chunks", len(chunks))
                with m_col2:
                    avg_size = sum(c.character_count for c in chunks) / len(chunks) if chunks else 0
                    st.metric("Avg Chunk Size", f"{avg_size:.1f} chars")
                with m_col3:
                    total_chars = sum(c.character_count for c in chunks)
                    st.metric("Total Characters", total_chars)
                with m_col4:
                    total_tokens = sum(c.token_estimate for c in chunks)
                    st.metric("Est. Total Tokens", total_tokens)
                
                # Interactive Table
                st.markdown("#### 📑 Chunks Table")
                table_data = []
                for c in chunks:
                    table_data.append({
                        "Chunk ID": c.chunk_id,
                        "Source Reference": c.metadata.get("source_reference", "N/A"),
                        "Size (chars)": c.character_count,
                        "Est. Tokens": c.token_estimate,
                        "Text Preview": c.text[:100] + "..." if len(c.text) > 100 else c.text
                    })
                st.dataframe(table_data, use_container_width=True)
                
                # Metadata Preview
                st.markdown("#### ⚙️ Chunk Metadata Inspection")
                selected_chunk_id = st.selectbox(
                    "Select a Chunk to inspect its full Metadata JSON:",
                    options=[c.chunk_id for c in chunks],
                    key=f"inspect_{name}"
                )
                if selected_chunk_id:
                    selected_chunk = next(c for c in chunks if c.chunk_id == selected_chunk_id)
                    st.json(selected_chunk.metadata)
