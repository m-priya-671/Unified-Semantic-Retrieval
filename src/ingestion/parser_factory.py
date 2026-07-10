from src.ingestion.text_parser import PDFParser, DocxParser

class ParserFactory:
    @staticmethod
    def get_parser(file_extension):
        # Normalize the extension to lowercase and ensure it starts with dot
        ext = file_extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
            
        if ext == ".pdf":
            return PDFParser()
        elif ext == ".docx":
            return DocxParser()
        elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            from src.image_processing.image_processor import ImageProcessor
            return ImageProcessor()
        elif ext in [".mp3", ".wav", ".m4a", ".flac"]:
            from src.audio_processing.audio_processor import AudioProcessor
            return AudioProcessor()
            
        raise ValueError(f"Unsupported file type: {file_extension}")