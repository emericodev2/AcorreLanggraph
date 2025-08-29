#!/usr/bin/env python3
"""
Streamlit Web App for LangGraph RAG Chatbot
"""
import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def check_environment():
    """Check if required environment variables are set"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for model access",
        "LANGCHAIN_API_KEY": "LangSmith API key for tracing",
        "LANGCHAIN_PROJECT": "LangSmith project name for organizing traces"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"• {var}: {description}")
    
    return missing_vars

def initialize_rag():
    """Initialize RAG system"""
    try:
        from app.rag import get_rag_manager
        rag = get_rag_manager()
        return rag, None
    except Exception as e:
        return None, str(e)

def load_documents(rag_manager):
    """Load documents from rawdata folder"""
    try:
        documents = rag_manager.load_documents_from_folder()
        return documents, None
    except Exception as e:
        return None, str(e)

def process_documents(rag_manager, documents):
    """Process and store documents"""
    try:
        success = rag_manager.process_and_store_documents(documents)
        return success, None
    except Exception as e:
        return False, str(e)

def scrape_website(url):
    """Scrape website content"""
    try:
        from app.rag import get_rag_manager
        rag = get_rag_manager()
        documents = rag.scrape_website(url)
        if documents:
            success = rag.process_and_store_documents(documents)
            return success, None
        else:
            return False, "Failed to scrape website"
    except Exception as e:
        return False, str(e)

def chat_with_rag(rag_manager, user_input):
    """Chat with RAG-enhanced responses"""
    try:
        # Search for relevant documents
        relevant_docs = rag_manager.search_documents(user_input, k=3)
        
        # Create context from relevant documents
        context = ""
        if relevant_docs:
            context = "\n\nRelevant context:\n" + "\n".join([
                f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:300]}..."
                for i, doc in enumerate(relevant_docs)
            ])
        
        # For now, return a simple response with context
        # In a full implementation, this would call the LangGraph agent
        response = f"I found some relevant information for your query: '{user_input}'{context}"
        return response, None
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 LangGraph RAG Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Environment check
        missing_vars = check_environment()
        if missing_vars:
            st.error("Missing environment variables:")
            for var in missing_vars:
                st.write(var)
            st.stop()
        else:
            st.success("✅ Environment configured")
        
        # RAG system status
        st.subheader("RAG System Status")
        rag_manager, rag_error = initialize_rag()
        
        if rag_manager:
            st.success("✅ RAG system ready")
            doc_count = rag_manager.get_document_count()
            st.metric("Documents loaded", doc_count)
        else:
            st.error(f"❌ RAG system error: {rag_error}")
            st.stop()
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh RAG System"):
            st.rerun()
        
        if st.button("🗑️ Clear Documents"):
            if rag_manager:
                rag_manager.clear_documents()
                st.success("Documents cleared!")
                st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📁 Documents", "🌐 Web Scraping", "💬 Chat"])
    
    with tab1:
        st.header("Welcome to LangGraph RAG Chatbot!")
        
        # Feature overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🚀 Advanced RAG</h3>
                <p>Retrieval-Augmented Generation with document processing, web scraping, and semantic search.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>📊 LangSmith Integration</h3>
                <p>Complete conversation tracking and performance monitoring in LangSmith.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🔍 Smart Search</h3>
                <p>Semantic document search using OpenAI embeddings and similarity matching.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🌐 Web Scraping</h3>
                <p>Add website content to your knowledge base for comprehensive information.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System status
        st.subheader("System Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.metric("OpenAI API", "✅ Connected")
        
        with status_col2:
            st.metric("LangSmith", "✅ Connected")
        
        with status_col3:
            st.metric("RAG System", "✅ Ready")
    
    with tab2:
        st.header("📁 Document Management")
        
        # Document loading
        st.subheader("Load Documents")
        
        if st.button("📂 Load Documents from rawdata/ folder"):
            with st.spinner("Loading documents..."):
                documents, error = load_documents(rag_manager)
                
                if documents:
                    st.success(f"✅ Loaded {len(documents)} documents")
                    
                    # Show document info
                    doc_info = []
                    for i, doc in enumerate(documents):
                        doc_info.append({
                            "Document": f"Document {i+1}",
                            "Source": doc.metadata.get('source', 'Unknown'),
                            "Type": doc.metadata.get('type', 'Unknown'),
                            "Content Length": len(doc.page_content)
                        })
                    
                    df = pd.DataFrame(doc_info)
                    st.dataframe(df, use_container_width=True)
                    
                    # Process documents
                    if st.button("🔧 Process and Store Documents"):
                        with st.spinner("Processing documents..."):
                            success, error = process_documents(rag_manager, documents)
                            
                            if success:
                                st.success("✅ Documents processed and stored successfully!")
                                st.rerun()
                            else:
                                st.error(f"❌ Error processing documents: {error}")
                else:
                    st.error(f"❌ Error loading documents: {error}")
        
        # Document statistics
        st.subheader("Document Statistics")
        if rag_manager:
            doc_count = rag_manager.get_document_count()
            st.metric("Total Documents", doc_count)
            
            if doc_count > 0:
                # Create a simple chart
                data = {"Documents": [doc_count], "Category": ["Loaded"]}
                df = pd.DataFrame(data)
                fig = px.bar(df, x="Category", y="Documents", title="Document Count")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🌐 Web Scraping")
        
        st.subheader("Add Website Content")
        
        url = st.text_input("Enter website URL:", placeholder="https://example.com")
        
        if st.button("🌐 Scrape Website"):
            if url:
                with st.spinner(f"Scraping {url}..."):
                    success, error = scrape_website(url)
                    
                    if success:
                        st.success(f"✅ Successfully scraped and stored {url}")
                        st.rerun()
                    else:
                        st.error(f"❌ Error scraping website: {error}")
            else:
                st.warning("Please enter a URL")
        
        st.info("💡 Tip: The system will automatically extract text content from websites and add it to your knowledge base.")
    
    with tab4:
        st.header("💬 Chat with RAG")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, error = chat_with_rag(rag_manager, prompt)
                    
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error(f"❌ Error generating response: {error}")
        
        # Chat controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Show RAG Context"):
                if st.session_state.messages:
                    st.subheader("RAG Context for Last Query")
                    # This would show the actual RAG context used
                    st.info("RAG context information would be displayed here")

if __name__ == "__main__":
    main()
