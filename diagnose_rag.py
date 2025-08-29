#!/usr/bin/env python3
"""
Diagnostic script for RAG system
"""
import os
from dotenv import load_dotenv

def diagnose_rag():
    """Diagnose RAG system issues"""
    print("🔍 RAG System Diagnosis")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment
    print("\n1. Environment Check:")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"   ✅ OpenAI API key: Found (length: {len(api_key)})")
    else:
        print("   ❌ OpenAI API key: Missing")
        return False
    
    # Check RAG manager
    print("\n2. RAG Manager Check:")
    try:
        from app.rag import get_rag_manager
        rag = get_rag_manager()
        print("   ✅ RAG manager: Created successfully")
    except Exception as e:
        print(f"   ❌ RAG manager: Failed to create - {e}")
        return False
    
    # Check vector store
    print("\n3. Vector Store Check:")
    if rag.vectorstore is not None:
        print("   ✅ Vector store: Initialized")
        print(f"   📁 Persist directory: {rag.persist_directory}")
        print(f"   📊 Document count: {rag.get_document_count()}")
    else:
        print("   ❌ Vector store: Not initialized")
        return False
    
    # Check rawdata folder
    print("\n4. Raw Data Folder Check:")
    from pathlib import Path
    rawdata_path = Path("rawdata")
    if rawdata_path.exists():
        files = list(rawdata_path.glob("*"))
        print(f"   ✅ Raw data folder: Found with {len(files)} items")
        for file in files:
            if file.is_file():
                print(f"      📄 {file.name}")
    else:
        print("   ⚠️  Raw data folder: Not found")
    
    # Test document loading
    print("\n5. Document Loading Test:")
    try:
        documents = rag.load_documents_from_folder()
        print(f"   ✅ Document loading: {len(documents)} documents loaded")
        
        if documents:
            print("\n6. Document Processing Test:")
            success = rag.process_and_store_documents(documents)
            if success:
                print("   ✅ Document processing: Successful")
                final_count = rag.get_document_count()
                print(f"   📊 Final document count: {final_count}")
            else:
                print("   ❌ Document processing: Failed")
                return False
        else:
            print("   ⚠️  No documents to process")
            
    except Exception as e:
        print(f"   ❌ Document loading/processing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Diagnosis completed successfully!")
    return True

if __name__ == "__main__":
    success = diagnose_rag()
    if not success:
        print("\n❌ Diagnosis found issues!")
