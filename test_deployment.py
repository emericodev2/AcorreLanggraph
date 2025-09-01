#!/usr/bin/env python3
"""
Test script to verify deployment structure.
Run this to ensure your code can be deployed to LangGraph platform.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import app
        print("  ✅ app package imported")
        
        from app import agent, state, tools, rag, rag_fallback
        print("  ✅ All app modules imported")
        
        # Test LangGraph specific imports
        from langgraph.graph import StateGraph
        print("  ✅ LangGraph imports work")
        
        from langchain_openai import ChatOpenAI
        print("  ✅ LangChain OpenAI imports work")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_agent_building():
    """Test that the agent can be built without errors"""
    print("\n🧪 Testing agent building...")
    
    try:
        # Set a dummy API key for testing
        os.environ["OPENAI_API_KEY"] = "test-key-for-deployment-test"
        
        from app.agent import build_agent
        
        graph = build_agent()
        print(f"  ✅ Agent built successfully! Graph type: {type(graph)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent building failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "langgraph.json",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "langgraph_app.py",
        "app/__init__.py",
        "app/agent.py",
        "app/state.py",
        "app/tools.py",
        "app/rag.py",
        "app/rag_fallback.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🚀 LangGraph Deployment Test Suite")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Agent Building", test_agent_building),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your code should deploy successfully.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
