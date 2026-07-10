# Offline Multimodal RAG Application

An offline, production-ready Multimodal Retrieval-Augmented Generation (RAG) system running entirely on Windows. This application enables ingestion, parsing, OCR, transcription, and semantic retrieval of heterogeneous documents (PDFs, Word documents, Images, Audios) with zero cloud dependencies, prioritizing complete privacy and local performance.

---

## 🚀 Features

### Milestone 1: Core Document Ingestion
- **PDF Page-by-Page Parser**: Custom text extraction utilizing PyMuPDF.
- **DOCX Block Parser**: Structured paragraph and table parser utilizing python-docx.
- **Robust Storage**: Automated local file system organization under `data/uploads/`.
- **Premium UI Dashboard**: Streamlit interface displaying file metrics, dynamic progress indicators, and text preview zones.

### Milestone 2: Multimodal Image & OCR Parser
- **OCR Engine Coordinator (`ocr_manager.py`)**: Selects and routes images to EasyOCR or Tesseract.
- **Image Preprocessor Pipeline**: Corrects contrast (CLAHE), grayscale conversion, noise reduction, thresholding, and auto-rotation (using Tesseract OSD).
- **Tamil & Multilingual Support**: Monkey-patched characters dictionary for EasyOCR Tamil engine compatibility.
- **Detailed Bounding Box & Coordinates View**: Renders segments and bounding boxes inside interactive DataFrames.

---

## 🛠️ Technologies Used

- **Framework**: Python 3.12, Streamlit
- **Document Extractors**: PyMuPDF, python-docx
- **Computer Vision & OCR**: EasyOCR, pytesseract (Tesseract OCR), OpenCV, Pillow
- **Deep Learning Execution**: PyTorch (CPU mode by default)
- **Log Management**: Standard library Logger with custom file rotating output.

---

## 🏗️ Project Architecture

```
                       [ Streamlit UI App (app.py) ]
                                     │
                        [ Ingestion / Parser Layer ]
                                     │
             ┌───────────────────────┼───────────────────────┐
             ▼                       ▼                       ▼
       [ PDF Parser ]         [ DOCX Parser ]        [ Image Processor ]
        (PyMuPDF)              (python-docx)                 │
                                                             ▼
                                                    [ OCR Manager Router ]
                                                             │
                                                   ┌─────────┴─────────┐
                                                   ▼                   ▼
                                              [ EasyOCR ]        [ Tesseract ]
```

---

## 📁 Folder Structure

```
Offline-RAG/
├── config/
│   └── settings.py          # Unified system and parser size configurations
├── data/
│   ├── database/            # SQLite storage (preserved by .gitkeep)
│   ├── indices/             # FAISS indexes (preserved by .gitkeep)
│   └── uploads/             # User uploaded files (preserved by .gitkeep)
├── docs/
│   └── verify_ui.md         # UI verification and manual checklist guides
├── logs/
│   └── app.log              # Local application error and trace logs (ignored)
├── models/
│   └── easyocr/             # Cached model weights (ignored)
├── src/
│   ├── config/
│   │   └── settings.py      # Base configurations
│   ├── image_processing/    # Validator, Preprocessor, OCR engines, Managers
│   ├── ingestion/           # PDF and Word parser subclasses
│   └── utils/               # File handlers and loggers
├── app.py                   # Main Streamlit web dashboard
├── upload_helper.py         # Playwright CDP automation script
├── verify_milestone1.py     # Milestone 1 automated tests
└── verify_milestone2.py     # Milestone 2 automated tests (OCR and boundaries)
```

---

## ⚙️ Installation

### Prerequisites
1. Install **Python 3.12** on Windows.
2. Install **Tesseract OCR** on Windows and ensure `tesseract.exe` is added to your system's PATH.

### Setup Instructions
1. Clone the repository locally:
   ```bash
   git clone https://github.com/your-username/Offline-RAG.git
   cd Offline-RAG
   ```
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install the dependencies listed in `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🖥️ Running the Application

Start the Streamlit application server:
```powershell
.\venv\Scripts\streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Roadmap (Milestones 1–8)

- [x] **Milestone 1**: Core Ingestion (PDF & DOCX parsing, Streamlit UI)
- [x] **Milestone 2**: Image Processing & OCR (EasyOCR/Tesseract routing, preprocessing pipeline)
- [x] **Milestone 3**: Audio Ingestion & Transcription (Whisper integration)
- [x] **Milestone 4**: Text Chunking Strategy (Semantic and hierarchical text splitter)
- [x] **Milestone 5**: Embeddings & Vector Stores (Local sentence-transformers)
- [ ] **Milestone 6**: FAISS Index Construction & Search
- [ ] **Milestone 7**: Context Retrieval Pipeline & Reranking
- [ ] **Milestone 8**: Local LLM Integration (Ollama local inference)

---

## 📷 Screenshots

*Placeholder for dashboard screenshots*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
