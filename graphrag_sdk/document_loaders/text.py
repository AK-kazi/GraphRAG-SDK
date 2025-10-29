import re
from typing import Iterator, Optional
from graphrag_sdk.document import Document

class TextLoader():
    """
    Load Text
    """

    def __init__(self, path: str, enable_streaming: bool = False, chunk_size: int = 8192, overlap: int = 100) -> None:
        """
        Initialize loader

        Args:
            path (str): path to Text.
            enable_streaming (bool): Enable streaming chunking for large files
            chunk_size (int): Size of chunks when streaming
            overlap (int): Overlap between chunks for context preservation
        """

        self.path = path
        self.enable_streaming = enable_streaming
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self) -> Iterator[Document]:
        """
        Load Text

        Returns:
            Iterator[Document]: document iterator
        """
        if self.enable_streaming:
            yield from self._load_streaming()
        else:
            yield from self._load_traditional()

    def _load_traditional(self) -> Iterator[Document]:
        """Traditional loading method for backward compatibility"""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                yield Document(
                    f.read(),
                    self.path
                )
        except UnicodeDecodeError:
            # Fallback to different encoding
            with open(self.path, 'r', encoding='latin-1') as f:
                yield Document(
                    f.read(),
                    self.path
                )

    def _load_streaming(self) -> Iterator[Document]:
        """Load text file in streaming chunks"""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                buffer = ""
                chunk_id = 0
                
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        if buffer.strip():
                            yield Document(buffer.strip(), f"{self.path}#final_{chunk_id}")
                        break
                    
                    buffer += chunk
                    if len(buffer) >= self.chunk_size + self.overlap:
                        # Find optimal breaking point
                        break_point = self._find_break_point(buffer)
                        content = buffer[:break_point].strip()
                        
                        if content:
                            yield Document(content, f"{self.path}#chunk_{chunk_id}")
                            chunk_id += 1
                        
                        buffer = buffer[break_point - self.overlap:]
                        
        except UnicodeDecodeError:
            # Fallback to different encoding
            with open(self.path, 'r', encoding='latin-1') as f:
                content = f.read()
                yield Document(content, self.path)
    
    def _find_break_point(self, text: str) -> int:
        """Find optimal breaking point in text"""
        # Prefer sentence breaks
        for i in range(len(text) - 1, max(0, len(text) - 500), -1):
            if text[i] in '.!?' and i < len(text) - 1 and text[i+1] == ' ':
                return i + 1
        
        # Then paragraph breaks
        for i in range(len(text) - 1, max(0, len(text) - 200), -1):
            if text[i] == '\n' and i > 0 and text[i-1] == '\n':
                return i + 1
        
        # Finally word breaks
        for i in range(len(text) - 1, max(0, len(text) - 100), -1):
            if text[i] == ' ':
                return i + 1
        
        return len(text) // 2
