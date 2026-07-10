from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Document:
    """Class representing a document chunk or page parsed from a file."""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata  # e.g., {'source': '...', 'page': 1, 'type': 'pdf'}

    def __repr__(self) -> str:
        return f"Document(text_len={len(self.text)}, metadata={self.metadata})"

class BaseParser(ABC):
    """Abstract Base Class for all document and media parsers in the system."""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parses a file and returns a list of Document objects.
        
        Args:
            file_path: The absolute or relative path to the file.
            
        Returns:
            A list of Document objects extracted from the file.
        """
        pass
