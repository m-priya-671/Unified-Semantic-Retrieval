# Streamlit UI Verification Procedure

This guide provides instructions to run and verify the Streamlit User Interface for the Offline Document Parser (Milestone 1).

---

## 1. Prerequisites
Ensure the virtual environment is set up and all required packages are installed (which is already configured in this workspace).

## 2. Launch the Application
Run the following command from the project root (`Offline-RAG/`) to launch the Streamlit development server:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Upon startup, the console will output local and network URLs:
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://<your-ip-address>:8501`

A browser window should automatically open to the application page. If it doesn't, navigate to `http://localhost:8501`.

---

## 3. UI Element Checklist

Verify that the following visual elements render correctly (complying with premium UI guidelines):
1. **Title Banner**: Displays a premium color gradient title reading **Unified Semantic Retrieval** and subtitle **Milestone 1: Offline PDF & DOCX Document Parser**.
2. **Sidebar Controls**:
   - Status indicators (e.g. `Offline State: Ready 🟢`).
   - Extraction stats showing `Total Files Indexed` and `Pages / Blocks Extracted`.
   - A red **Clear Parsing Memory** button.
3. **Ingestion Zone**: A drag-and-drop file uploader area with support for `.pdf` and `.docx` files.

---

## 4. Interaction Test Matrix

| Step | Action | Expected Behavior |
| :--- | :--- | :--- |
| **1** | Try dragging a non-supported file (e.g., an `.xlsx` or `.txt` file) into the uploader. | The uploader should block or show a warning. If uploaded, the uploader displays an error indicating the format is not allowed. |
| **2** | Drag and drop the generated PDF test file: `data/uploads/sample_test.pdf`. | - A progress indicator appears showing validation and parsing progress.<br>- A green success message appears: `Parsed sample_test.pdf (x KB) - Extracted 2 structures.` |
| **3** | Expand the file detail view by clicking **Show Extracted Content (2 Blocks/Pages)**. | - You should see two headers: **Page 1 / 2** and **Page 2 / 2**.<br>- Under each page, a text box displays extracted text content (e.g., "Offline Multimodal RAG Project...") and metadata as JSON. |
| **4** | Drag and drop the generated DOCX test file: `data/uploads/sample_test.docx`. | - A progress indicator displays and completes successfully.<br>- A green success message appears showing blocks count. |
| **5** | Expand **Show Extracted Content (4 Blocks/Pages)** under `sample_test.docx`. | - Block 3 shows extracted Tamil text: `"தமிழ் உரை சோதனை:..."`.<br>- Block 4 shows the parsed table in structured text layout: `"Language \| Greeting \n English \| Hello \n Tamil \| வணக்கம்"`. |
| **6** | Verify the Sidebar statistics update. | - Total Files Indexed increments to `2`.<br>- Pages / Blocks Extracted updates to `6` (2 pages from PDF + 4 blocks from DOCX). |
| **7** | Click **Clear Parsing Memory** in the sidebar. | - The session state resets.<br>- Ingested documents list becomes empty.<br>- Statistics reset to `0`. |
