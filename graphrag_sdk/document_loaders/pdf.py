from typing import Iterator
from graphrag_sdk.document import Document

class PDFLoader():
    """
    Load PDF
    """

    def __init__(self, path: str, enable_parallel: bool = False, max_workers: int = 4) -> None:
        """
        Initialize loader

        Args:
            path (str): path to PDF.
            enable_parallel (bool): Enable parallel page processing
            max_workers (int): Maximum number of worker threads for parallel processing
        """

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )

        self.path = path
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

    def load(self) -> Iterator[Document]:
        """
        Load PDF

        Returns:
            Iterator[Document]: document iterator
        """
        if self.enable_parallel:
            yield from self._load_parallel()
        else:
            yield from self._load_traditional()

    def _load_traditional(self) -> Iterator[Document]:
        """Traditional loading method for backward compatibility"""
        from pypdf import PdfReader # pylint: disable=import-outside-toplevel

        reader = PdfReader(self.path)
        yield from [
            Document(page_content.extract_text(), f"{self.path}#{page_num}")
            for page_num, page_content in enumerate(reader.pages)
        ]

    def _load_parallel(self) -> Iterator[Document]:
        """Load PDF with parallel page processing"""
        try:
            from pypdf import PdfReader
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with open(self.path, 'rb') as file:
                pdf_reader = PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                def extract_page(page_num):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    return Document(text, f"{self.path}#page_{page_num + 1}")
                
                # Process pages in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(extract_page, i) for i in range(total_pages)]
                    
                    for future in as_completed(futures):
                        try:
                            doc = future.result()
                            if doc.not_empty():
                                yield doc
                        except Exception as e:
                            print(f"Error processing PDF page: {e}")
                            
        except Exception as e:
            raise ValueError(f"Failed to load PDF {self.path}: {e}")
