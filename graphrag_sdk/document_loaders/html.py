import re
import requests
from typing import Iterator
from graphrag_sdk.document import Document

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class HTMLLoader:
    """
    Load HTML
    """

    def __init__(self, path: str, enable_streaming: bool = False, chunk_size: int = 8192) -> None:
        """
        Initialize loader

        Args:
            path (str): path to HTML.
            enable_streaming (bool): Enable streaming chunking for large files
            chunk_size (int): Size of chunks when streaming
        """

        self.path = path
        self.enable_streaming = enable_streaming
        self.chunk_size = chunk_size

    def _get_file(self) -> str:
        try:
            with open(self.path, "r") as f:
                return f.read()
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"An error occurred: {e}")
            return ""

    def load(self) -> Iterator[Document]:
        """
        Load HTML

        Returns:
            Iterator[Document]: document iterator
        """
        if self.enable_streaming:
            yield from self._load_streaming()
        else:
            yield from self._load_traditional()

    def _load_traditional(self) -> Iterator[Document]:
        """Traditional loading method for backward compatibility"""
        # Download URL
        content = self._get_file()

        if BS4_AVAILABLE:
            # extract text from HTML, populate content
            soup = BeautifulSoup(content, "html.parser")

            # Extract text from the HTML
            content = soup.get_text()

            # Remove extra newlines
            content = re.sub(r"\n{2,}", "\n", content)
        else:
            # Fallback if BeautifulSoup is not available
            print("BeautifulSoup not available, using raw HTML content")
        
        yield Document(content, self.path)

    def _load_streaming(self) -> Iterator[Document]:
        """Load and parse HTML in streaming fashion"""
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize newlines
                text = re.sub(r' {2,}', ' ', text)  # Normalize spaces
            else:
                # Fallback if BeautifulSoup is not available
                text = content
                print("BeautifulSoup not available, using raw HTML content")
            
            # Split into chunks
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    yield Document(chunk.strip(), f"{self.path}#chunk_{i//self.chunk_size}")
                    
        except Exception as e:
            # Fallback to plain text
            with open(self.path, 'r', encoding='utf-8') as f:
                content = f.read()
                yield Document(content, self.path)