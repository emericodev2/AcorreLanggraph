#!/usr/bin/env python3
"""
Minimal test for ChromaDB initialization
"""
import os
from dotenv import load_dotenv

def test_chromadb():
    """Test ChromaDB initialization directly"""
    print("Testing ChromaDB initialization...")
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Test OpenAI embeddings
        from langchain_openai import OpenAIEmbeddings
        print("✅ OpenAI embeddings imported")
        
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        print("✅ OpenAI embeddings created")
        
        # Test ChromaDB
        from langchain_community.vectorstores import Chroma
        print("✅ ChromaDB imported")
        
        # Test vector store creation
        persist_dir = "test_chroma"
        print(f"🔧 Creating test vector store at {persist_dir}")
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("✅ ChromaDB vector store created")
        
        # Test basic operations
        test_count = vectorstore._collection.count()
        print(f"✅ Vector store test successful, count: {test_count}")
        
        # Clean up
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            print("✅ Test directory cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chromadb()
    if success:
        print("\n🎉 ChromaDB test successful!")
    else:
        print("\n❌ ChromaDB test failed!")
