#!/usr/bin/env python3
"""
Test script to verify RAG document processing and storage
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rag_processing():
    """Test the complete RAG document processing pipeline"""
    print("🚀 Testing RAG Document Processing Pipeline...")
    
    try:
        # Import and initialize RAG manager
        from app.rag import get_rag_manager
        rag_manager = get_rag_manager()
        print(f"✅ RAG Manager initialized: {type(rag_manager).__name__}")
        
        # Check initial state
        initial_count = rag_manager.get_document_count()
        print(f"📊 Initial document count: {initial_count}")
        
        # Load documents
        print("\n📂 Loading documents...")
        documents = rag_manager.load_documents_from_folder()
        print(f"📄 Loaded {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            print(f"  Document {i+1}: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Content length: {len(doc.page_content)}")
            print(f"    Type: {doc.metadata.get('type', 'Unknown')}")
        
        # Process and store documents
        print("\n🔧 Processing and storing documents...")
        success = rag_manager.process_and_store_documents(documents)
        print(f"✅ Processing result: {success}")
        
        if success:
            # Check final state
            final_count = rag_manager.get_document_count()
            print(f"📊 Final document count: {final_count}")
            print(f"📈 Documents added: {final_count - initial_count}")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            test_queries = ["RAG functionality", "automotive", "ASD"]
            
            for query in test_queries:
                print(f"  Searching for: '{query}'")
                results = rag_manager.search_documents(query, k=2)
                print(f"    Found {len(results)} results")
                if results:
                    for i, doc in enumerate(results):
                        source = doc.metadata.get('source', 'Unknown')
                        preview = doc.page_content[:100].replace('\n', ' ')
                        print(f"      Result {i+1}: {source} - {preview}...")
            
            print("\n🎉 RAG processing test completed successfully!")
            return True
        else:
            print("❌ Document processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_processing()
    if success:
        print("\n✅ All tests passed! The RAG system is working correctly.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
