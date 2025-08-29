#!/usr/bin/env python3
"""
Startup script for LangGraph RAG Chatbot
Choose between CLI and Web UI versions
"""
import os
import sys
from pathlib import Path

def check_environment():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    
    # Try to load .env file
    env_loaded = load_dotenv()
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for model access",
        "LANGCHAIN_API_KEY": "LangSmith API key for tracing",
        "LANGCHAIN_PROJECT": "LangSmith project name for organizing traces"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"• {var}: {description}")
    
    return missing_vars, env_loaded

def main():
    print("🤖 LangGraph RAG Chatbot")
    print("=" * 50)
    
    # Check environment
    print("🔍 Checking environment...")
    missing_vars, env_loaded = check_environment()
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        
        if not env_loaded:
            print("\n💡 No .env file found. Please create one with:")
            print("OPENAI_API_KEY=your_openai_key")
            print("LANGCHAIN_API_KEY=your_langsmith_key")
            print("LANGCHAIN_PROJECT=your_project_name")
        else:
            print("\n💡 Please check your .env file and ensure all required variables are set.")
        
        print("\n🚀 You can still try to run the system if you have these set in your environment.")
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            return
    else:
        print("✅ Environment configured successfully!")
    
    # Check if Streamlit is available
    try:
        import streamlit
        streamlit_available = True
    except ImportError:
        streamlit_available = False
        print("⚠️  Streamlit not available. Install with: pip install streamlit")
    
    # Show options
    print("\n🚀 Choose your interface:")
    print("1. 🌐 Web UI (Streamlit) - Beautiful, interactive interface")
    print("2. 💻 CLI - Command-line interface")
    print("3. 🔧 Install Web UI dependencies")
    print("4. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            if not streamlit_available:
                print("❌ Streamlit not available. Please install it first (option 3).")
                continue
            
            print("🚀 Starting Web UI...")
            print("📱 The web interface will open in your browser.")
            print("💡 To stop the server, press Ctrl+C in this terminal.")
            
            # Start Streamlit app
            os.system(f"{sys.executable} -m streamlit run web_app_enhanced.py --server.port 8501")
            break
            
        elif choice == "2":
            print("🚀 Starting CLI...")
            print("💡 Type 'exit' to quit the CLI.")
            
            # Start CLI
            os.system(f"{sys.executable} cli.py")
            break
            
        elif choice == "3":
            print("🔧 Installing Web UI dependencies...")
            os.system(f"{sys.executable} -m pip install streamlit streamlit-chat plotly pandas numpy")
            print("✅ Web UI dependencies installed!")
            print("🔄 Please restart this script to use the Web UI.")
            break
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
