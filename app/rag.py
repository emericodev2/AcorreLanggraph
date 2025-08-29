import os
import requests
from typing import List, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader, 
    PDFMinerLoader, 
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGManager:
    """Manages RAG operations including document loading, web scraping, and vector storage"""
    
    def __init__(self, rawdata_folder: str = "rawdata", persist_directory: str = "chroma_db"):
        # Load environment variables first
        load_dotenv()
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for RAG functionality")
        
        self.rawdata_folder = Path(rawdata_folder)
        self.persist_directory = persist_directory
        
        # Initialize embeddings with explicit API key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"  # Use a specific embedding model
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create rawdata folder if it doesn't exist
        self.rawdata_folder.mkdir(exist_ok=True)
        
        # For now, use simple RAG to ensure the system works
        print("🔧 Initializing RAG system...")
        self.use_simple_rag = True
        self._initialize_simple_rag()
    
    def _initialize_simple_rag(self):
        """Initialize simple RAG as the primary system"""
        try:
            from app.rag_fallback import SimpleRAGManager
            self.simple_rag = SimpleRAGManager(str(self.rawdata_folder))
            print("✅ Simple RAG system initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize simple RAG: {e}")
            raise RuntimeError("Failed to initialize RAG system")
    
    def load_documents_from_folder(self) -> List[Document]:
        """Load all supported documents from the rawdata folder"""
        return self.simple_rag.load_documents_from_folder()
    
    def scrape_website(self, url: str) -> List[Document]:
        """Scrape content from a website and convert to documents"""
        try:
            print(f"🌐 Scraping website: {url}")
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = f"https://{url}"
            
            # Fetch webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "source": url,
                    "title": soup.title.string if soup.title else url,
                    "type": "website"
                }
            )
            
            print(f"  ✅ Successfully scraped {url}")
            return [doc]
            
        except Exception as e:
            print(f"  ❌ Failed to scrape {url}: {e}")
            return []
    
    def process_and_store_documents(self, documents: List[Document]) -> bool:
        """Process documents and store them in the vector store"""
        return self.simple_rag.process_and_store_documents(documents)
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents based on a query"""
        return self.simple_rag.search_documents(query, k)
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store"""
        return self.simple_rag.get_document_count()
    
    def clear_vectorstore(self):
        """Clear all documents from the vector store"""
        return self.simple_rag.clear_documents()


# Global RAG manager instance
rag_manager = None


def get_rag_manager() -> RAGManager:
    """Get or create the global RAG manager instance"""
    global rag_manager
    if rag_manager is None:
        rag_manager = RAGManager()
    return rag_manager
