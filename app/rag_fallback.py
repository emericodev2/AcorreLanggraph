import os
import re
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class SimpleRAGManager:
    """Simple RAG manager that works without external vector stores"""
    
    def __init__(self, rawdata_folder: str = "rawdata"):
        # Load environment variables first
        load_dotenv()
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for RAG functionality")
        
        self.rawdata_folder = Path(rawdata_folder)
        
        # Initialize embeddings with explicit API key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Create rawdata folder if it doesn't exist
        self.rawdata_folder.mkdir(exist_ok=True)
        
        # Simple in-memory storage
        self.documents = []
        self.document_embeddings = []
        
        print("✅ Simple RAG manager initialized (no external vector store required)")
    
    def load_documents_from_folder(self) -> List[Document]:
        """Load all supported documents from the rawdata folder"""
        documents = []
        supported_extensions = {
            '.txt': self._load_text_file,
            '.md': self._load_text_file,
        }
        
        if not self.rawdata_folder.exists():
            print(f"📁 Raw data folder '{self.rawdata_folder}' doesn't exist. Creating it...")
            self.rawdata_folder.mkdir(exist_ok=True)
            return documents
        
        print(f"📁 Loading documents from '{self.rawdata_folder}'...")
        
        for file_path in self.rawdata_folder.rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_ext in supported_extensions:
                    try:
                        docs = supported_extensions[file_ext](file_path)
                        documents.extend(docs)
                        print(f"  ✅ Loaded {file_path.name} ({len(docs)} chunks)")
                    except Exception as e:
                        print(f"  ❌ Failed to load {file_path.name}: {e}")
                else:
                    print(f"  ⚠️  Skipped {file_path.name} (unsupported format)")
        
        print(f"📊 Total documents loaded: {len(documents)}")
        return documents
    
    def _load_text_file(self, file_path: Path) -> List[Document]:
        """Load text files (txt, md)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "text_file"
                }
            )
            return [doc]
        except Exception as e:
            print(f"    Error reading {file_path}: {e}")
            return []
    
    def process_and_store_documents(self, documents: List[Document]) -> bool:
        """Process documents and store them in memory"""
        if not documents:
            print("⚠️  No documents to process")
            return False
        
        try:
            print(f"🔧 Processing {len(documents)} documents...")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"  📝 Created {len(chunks)} text chunks")
            
            # Store chunks in memory
            self.documents = chunks
            print(f"  💾 Stored {len(chunks)} chunks in memory")
            
            # Generate embeddings for chunks
            print("  🔍 Generating embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            self.document_embeddings = self.embeddings.embed_documents(texts)
            print(f"  ✅ Generated embeddings for {len(chunks)} chunks")
            
            return True
                
        except Exception as e:
            print(f"  ❌ Error processing documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents using simple similarity"""
        if not self.documents or not self.document_embeddings:
            print("❌ No documents loaded for search")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Simple cosine similarity
            similarities = []
            for i, doc_embedding in enumerate(self.document_embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and get top k
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            results = [self.documents[idx] for idx in top_indices]
            print(f"🔍 Found {len(results)} relevant documents for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            # Fallback to simple dot product if numpy not available
            return sum(a * b for a, b in zip(vec1, vec2))
    
    def get_document_count(self) -> int:
        """Get the total number of documents in memory"""
        return len(self.documents)
    
    def clear_documents(self):
        """Clear all documents from memory"""
        self.documents = []
        self.document_embeddings = []
        print("🗑️  Cleared all documents from memory")


def get_simple_rag_manager() -> SimpleRAGManager:
    """Get or create a simple RAG manager instance"""
    return SimpleRAGManager()
