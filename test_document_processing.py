#!/usr/bin/env python3
"""
Test document processing functionality
"""
import os
from dotenv import load_dotenv

def test_document_processing():
    """Test document loading and processing"""
    print("Testing Document Processing...")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Import and test RAG manager
        from app.rag import get_rag_manager
        print("✅ RAG module imported successfully")
        
        rag = get_rag_manager()
        print("✅ RAG manager created successfully")
        
        # Test document loading
        print("\n📁 Testing document loading...")
        documents = rag.load_documents_from_folder()
        print(f"✅ Loaded {len(documents)} documents")
        
        if documents:
            # Test document processing
            print("\n🔧 Testing document processing...")
            success = rag.process_and_store_documents(documents)
            
            if success:
                print("✅ Document processing successful!")
                doc_count = rag.get_document_count()
                print(f"📊 Vector store now contains {doc_count} chunks")
            else:
                print("❌ Document processing failed")
        else:
            print("⚠️  No documents found to process")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_document_processing()
    if success:
        print("\n🎉 Document processing test completed!")
    else:
        print("\n❌ Document processing test failed!")
