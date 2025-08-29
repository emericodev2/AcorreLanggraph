#!/usr/bin/env python3
"""
Debug script to test RAG functionality step by step
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY found: {openai_key[:20]}...")
    else:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    return True

def test_rag_import():
    """Test RAG module import"""
    print("\n🔍 Testing RAG module import...")
    
    try:
        from app.rag import get_rag_manager
        print("✅ RAG module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import RAG module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_manager_creation():
    """Test RAG manager creation"""
    print("\n🔍 Testing RAG manager creation...")
    
    try:
        from app.rag import get_rag_manager
        rag_manager = get_rag_manager()
        print("✅ RAG manager created successfully")
        return rag_manager
    except Exception as e:
        print(f"❌ Failed to create RAG manager: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_document_loading(rag_manager):
    """Test document loading"""
    print("\n🔍 Testing document loading...")
    
    try:
        documents = rag_manager.load_documents_from_folder()
        print(f"✅ Loaded {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            print(f"  Document {i+1}: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Content length: {len(doc.page_content)}")
            print(f"    Type: {doc.metadata.get('type', 'Unknown')}")
        
        return documents
    except Exception as e:
        print(f"❌ Failed to load documents: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_document_processing(rag_manager, documents):
    """Test document processing"""
    print("\n🔍 Testing document processing...")
    
    try:
        success = rag_manager.process_and_store_documents(documents)
        if success:
            print("✅ Documents processed and stored successfully")
            doc_count = rag_manager.get_document_count()
            print(f"  Total documents in storage: {doc_count}")
            return True
        else:
            print("❌ Document processing failed")
            return False
    except Exception as e:
        print(f"❌ Failed to process documents: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_search(rag_manager):
    """Test document search"""
    print("\n🔍 Testing document search...")
    
    try:
        query = "RAG functionality"
        results = rag_manager.search_documents(query, k=3)
        print(f"✅ Search completed, found {len(results)} results")
        
        for i, doc in enumerate(results):
            print(f"  Result {i+1}: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Content preview: {doc.page_content[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Failed to search documents: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting RAG functionality debug...")
    
    # Test 1: Environment
    if not test_environment():
        print("\n❌ Environment test failed. Please check your .env file.")
        return
    
    # Test 2: RAG import
    if not test_rag_import():
        print("\n❌ RAG import test failed.")
        return
    
    # Test 3: RAG manager creation
    rag_manager = test_rag_manager_creation()
    if not rag_manager:
        print("\n❌ RAG manager creation failed.")
        return
    
    # Test 4: Document loading
    documents = test_document_loading(rag_manager)
    if not documents:
        print("\n❌ Document loading failed.")
        return
    
    # Test 5: Document processing
    if not test_document_processing(rag_manager, documents):
        print("\n❌ Document processing failed.")
        return
    
    # Test 6: Document search
    if not test_document_search(rag_manager):
        print("\n❌ Document search failed.")
        return
    
    print("\n🎉 All RAG functionality tests passed!")
    print("\nIf you're still having issues in the Web UI, the problem might be:")
    print("1. Streamlit session state management")
    print("2. RAG manager instance persistence between interactions")
    print("3. Error handling in the Web UI")

if __name__ == "__main__":
    main()
