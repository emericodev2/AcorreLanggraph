#!/usr/bin/env python3
"""
Test script for RAG functionality
"""
import os
from dotenv import load_dotenv

def test_rag():
    """Test RAG manager initialization"""
    print("Testing RAG Manager...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    print(f"✅ OpenAI API key found (length: {len(api_key)})")
    
    try:
        # Import and test RAG manager
        from app.rag import get_rag_manager
        print("✅ RAG module imported successfully")
        
        rag = get_rag_manager()
        print("✅ RAG manager created successfully")
        
        print(f"✅ Vector store initialized: {rag.vectorstore is not None}")
        print(f"📊 Document count: {rag.get_document_count()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG: {e}")
        return False

if __name__ == "__main__":
    success = test_rag()
    if success:
        print("\n🎉 RAG test completed successfully!")
    else:
        print("\n❌ RAG test failed!")
