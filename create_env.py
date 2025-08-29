#!/usr/bin/env python3
"""
Script to create .env file with environment variables
"""
import os

def create_env_file():
    """Create .env file with environment variables"""
    
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=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

# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=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
LANGCHAIN_PROJECT=AcorreLangchain

# OpenAI Model Configuration
OPENAI_MODEL=gpt-4o-mini
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("📝 Environment variables configured:")
        print("   - OpenAI API Key")
        print("   - LangSmith API Key")
        print("   - LangSmith Project")
        print("   - OpenAI Model")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Creating .env file...")
    if create_env_file():
        print("\n🚀 You can now run the chatbot!")
        print("   python start_web_app.py")
    else:
        print("\n❌ Failed to create .env file. Please create it manually.")
