#!/usr/bin/env python3
"""
Test script to verify LangSmith tracing integration
"""
import os
from dotenv import load_dotenv
from langsmith import Client

def test_langsmith_connection():
    """Test if we can connect to LangSmith and read the project"""
    load_dotenv()
    
    # Check environment variables
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT")
    
    print(f"LangSmith Configuration:")
    print(f"  API Key: {'✓ Set' if api_key else '✗ Missing'}")
    print(f"  Project: {project or '✗ Missing'}")
    print(f"  Endpoint: {endpoint or '✗ Missing'}")
    
    if not all([api_key, project, endpoint]):
        print("\n❌ Missing required environment variables!")
        return False
    
    try:
        # Test connection
        client = Client(
            api_url=endpoint,
            api_key=api_key
        )
        
        # Try to read the project
        project_info = client.read_project(project)
        print(f"\n✅ Successfully connected to LangSmith!")
        print(f"  Project: {project_info.name}")
        print(f"  Description: {project_info.description or 'No description'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to connect to LangSmith: {e}")
        return False

def test_openai_key():
    """Test if OpenAI API key is set"""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API Key: Set (length: {len(api_key)})")
        return True
    else:
        print("❌ OpenAI API Key: Missing")
        return False

if __name__ == "__main__":
    print("Testing LangSmith Integration...\n")
    
    openai_ok = test_openai_key()
    langsmith_ok = test_langsmith_connection()
    
    print("\n" + "="*50)
    if openai_ok and langsmith_ok:
        print("🎉 All tests passed! You can now run the chatbot with:")
        print("   .\\.venv\\Scripts\\python cli.py")
    else:
        print("❌ Some tests failed. Please check your .env file.")
        print("   Required variables:")
        print("   - OPENAI_API_KEY")
        print("   - LANGCHAIN_API_KEY") 
        print("   - LANGCHAIN_PROJECT")
        print("   - LANGCHAIN_ENDPOINT (optional, defaults to https://api.smith.langchain.com)")
