import re
import requests
import asyncio
from typing import Iterator, List, Optional, Union, AsyncGenerator
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

from graphrag_sdk.document import Document

class URLLoader():
    """
    Load URL
    """

    def __init__(self, url: str) -> None:
        """
        Initialize loader

        Args:
            url (str): url.
        """

        self.url = url

    def _download(self) -> Optional[str]:
        try:
            response = requests.get(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    def load(self) -> Iterator[Document]:
        """
        Load URL

        Returns:
            Iterator[Document]: document iterator
        """

        # Download URL
        content = self._download()
        
        if content is None:
            return

        # extract text from HTML, populate content
        if BeautifulSoup is None:
            # Fallback if BeautifulSoup is not available
            yield Document(content, self.url)
            return
            
        soup = BeautifulSoup(content, 'html.parser')

        # Extract text from the HTML
        content = soup.get_text()

        # Remove extra newlines
        content = re.sub(r'\n{2,}', '\n', content)

        yield Document(content, self.url)


class AsyncURLLoader:
    """
    Async URL Loader with connection pooling and retry logic
    """

    def __init__(self, urls: List[str], max_concurrent: int = 10, timeout: int = 30):
        """
        Initialize async URL loader

        Args:
            urls (List[str]): List of URLs to load
            max_concurrent (int): Maximum concurrent requests
            timeout (int): Request timeout in seconds
        """
        if aiohttp is None:
            raise ImportError("aiohttp is required for AsyncURLLoader. Install with: pip install aiohttp")
        if BeautifulSoup is None:
            raise ImportError("beautifulsoup4 is required for AsyncURLLoader. Install with: pip install beautifulsoup4")
            
        self.urls = urls
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def load(self) -> AsyncGenerator[Document, None]:
        """
        Load URLs asynchronously with connection pooling

        Returns:
            AsyncGenerator[Document, None]: async document iterator
        """
        # Configure connection pooling
        TCPConnector = getattr(aiohttp, 'TCPConnector', None)
        ClientTimeout = getattr(aiohttp, 'ClientTimeout', None)
        ClientSession = getattr(aiohttp, 'ClientSession', None)
        ClientError = getattr(aiohttp, 'ClientError', Exception)
        
        if TCPConnector is None or ClientTimeout is None or ClientSession is None:
            raise ImportError("aiohttp is required for AsyncURLLoader. Install with: pip install aiohttp")
            
        connector = TCPConnector(
            limit=50,  # Total connection pool size
            limit_per_host=10,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = ClientTimeout(total=self.timeout)
        
        headers = {
            'User-Agent': 'GraphRAG-SDK/1.0 (Ingestion Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        async with ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            # Create tasks for all URLs
            tasks = [self._fetch_url(session, url) for url in self.urls]
            
            # Process results as they complete
            for completed_task in asyncio.as_completed(tasks):
                try:
                    document = await completed_task
                    if document:
                        yield document
                except Exception as e:
                    print(f"Error fetching URL: {e}")
                    continue
    
    async def _fetch_url(self, session, url: str) -> Optional[Document]:
        """
        Fetch single URL with retry logic

        Args:
            session (aiohttp.ClientSession): HTTP session
            url (str): URL to fetch

        Returns:
            Optional[Document]: Document if successful, None otherwise
        """
        async with self.semaphore:
            max_retries = 3
            base_delay = 1
            
            for attempt in range(max_retries):
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' not in content_type:
                            print(f"Skipping non-HTML content: {content_type}")
                            return None
                        
                        # Read content with size limit
                        content = await response.text()
                        
                        # Extract text from HTML
                        text = self._extract_text_from_html(content)
                        
                        if text.strip():
                            return Document(text, url)
                        else:
                            return None
                            
                except Exception as e:
                    if "ClientError" in str(type(e)) or "aiohttp" in str(type(e)):
                        if attempt == max_retries - 1:
                            raise
                        delay = base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        print(f"Unexpected error fetching {url}: {e}")
                        return None
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract clean text from HTML content

        Args:
            html_content (str): Raw HTML content

        Returns:
            str: Cleaned text content
        """
        try:
            if BeautifulSoup is None:
                return html_content
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get text and normalize whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return html_content