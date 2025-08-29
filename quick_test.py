#!/usr/bin/env python3
"""
Quick test for RAG system functionality
"""
import os
from dotenv import load_dotenv

def quick_test():
    """Quick test of RAG system"""
    print("🚀 Quick RAG System Test")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Test basic imports
        from app.rag import get_rag_manager
        print("✅ RAG module imported successfully")
        
        # Test RAG manager creation
        rag = get_rag_manager()
        print("✅ RAG manager created successfully")
        
        # Test document loading (without processing)
        print("\n📁 Testing document loading...")
        documents = rag.load_documents_from_folder()
        print(f"✅ Loaded {len(documents)} documents")
        
        # Show document info
        for i, doc in enumerate(documents):
            print(f"  📄 Document {i+1}: {doc.metadata.get('source', 'Unknown')}")
            print(f"     Content preview: {doc.page_content[:100]}...")
        
        print(f"\n🎉 Quick test completed successfully!")
        print(f"📊 Documents ready for processing: {len(documents)}")
        print(f"🔧 RAG system status: {'Ready' if len(documents) > 0 else 'No documents'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n❌ Quick test failed!")
