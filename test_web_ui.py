#!/usr/bin/env python3
"""
Test script for Web UI components
"""
import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing Web UI imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_rag_system():
    """Test if RAG system can be imported"""
    print("\n🧪 Testing RAG system...")
    
    try:
        from app.rag import get_rag_manager
        print("✅ RAG module imported successfully")
        
        # Try to create RAG manager
        rag = get_rag_manager()
        print("✅ RAG manager created successfully")
        
        # Check document count
        doc_count = rag.get_document_count()
        print(f"✅ Document count: {doc_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_agent_system():
    """Test if agent system can be imported"""
    print("\n🧪 Testing agent system...")
    
    try:
        from app.agent import build_agent
        print("✅ Agent module imported successfully")
        
        # Try to build agent
        agent = build_agent()
        print("✅ Agent built successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent system test failed: {e}")
        return False

def main():
    print("🚀 Web UI Component Test")
    print("=" * 40)
    
    # Load environment
    load_dotenv()
    
    # Test imports
    if not test_imports():
        print("\n❌ Web UI dependencies not available")
        print("💡 Install with: pip install streamlit streamlit-chat plotly pandas numpy")
        return
    
    # Test RAG system
    if not test_rag_system():
        print("\n❌ RAG system not working")
        return
    
    # Test agent system
    if not test_agent_system():
        print("\n❌ Agent system not working")
        return
    
    print("\n🎉 All tests passed! Web UI is ready to run.")
    print("\n🚀 You can now run:")
    print("  python start_web_app.py")
    print("  or")
    print("  streamlit run web_app_enhanced.py")

if __name__ == "__main__":
    main()
