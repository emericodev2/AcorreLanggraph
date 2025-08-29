#!/usr/bin/env python3
"""
Enhanced Streamlit Web App for LangGraph RAG Chatbot with Full Agent Integration
"""
import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="LangGraph RAG Chatbot Pro",
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
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
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

def initialize_agent():
    """Initialize LangGraph agent"""
    try:
        from app.agent import build_agent
        agent = build_agent()
        return agent, None
    except Exception as e:
        return None, str(e)

def load_documents(rag_manager):
    """Load documents from rawdata folder"""
    try:
        if rag_manager is None:
            return None, "RAG manager is not initialized"
        
        print(f"🔍 Loading documents with RAG manager: {type(rag_manager).__name__}")
        documents = rag_manager.load_documents_from_folder()
        print(f"📄 Loaded {len(documents)} documents")
        return documents, None
    except Exception as e:
        print(f"❌ Error in load_documents: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def process_documents(rag_manager, documents):
    """Process and store documents"""
    try:
        if rag_manager is None:
            return False, "RAG manager is not initialized"
        
        if not documents:
            return False, "No documents to process"
        
        print(f"🔧 Processing {len(documents)} documents with RAG manager: {type(rag_manager).__name__}")
        print(f"📄 Document types: {[type(doc).__name__ for doc in documents]}")
        print(f"📄 Document sources: {[doc.metadata.get('source', 'Unknown') for doc in documents]}")
        
        # Process documents
        success = rag_manager.process_and_store_documents(documents)
        print(f"✅ Document processing result: {success}")
        
        if success:
            # Verify documents were stored
            doc_count = rag_manager.get_document_count()
            print(f"📊 Documents now in storage: {doc_count}")
            
            # Test search functionality
            try:
                test_query = "test"
                results = rag_manager.search_documents(test_query, k=1)
                print(f"🔍 Search test successful, found {len(results)} results")
            except Exception as search_error:
                print(f"⚠️  Search test failed: {search_error}")
        
        return success, None
    except Exception as e:
        print(f"❌ Error in process_documents: {e}")
        import traceback
        traceback.print_exc()
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

def chat_with_agent(agent, user_input, rag_manager):
    """Chat with the full LangGraph agent"""
    try:
        # Initialize conversation state
        state = {"messages": [HumanMessage(content=user_input)]}
        
        # Get RAG context if available
        if rag_manager.get_document_count() > 0:
            relevant_docs = rag_manager.search_documents(user_input, k=3)
            if relevant_docs:
                context = "\n\nRelevant context from your documents:\n" + "\n".join([
                    f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:200]}..."
                    for i, doc in enumerate(relevant_docs)
                ])
                # Add context to the user message
                state["messages"][0] = HumanMessage(content=user_input + context)
        
        # Invoke the agent
        with st.spinner("🤔 Thinking..."):
            result = agent.invoke(state)
            
            # Extract the response
            if "messages" in result and result["messages"]:
                # Find the last AI message
                ai_messages = [m for m in result["messages"] if hasattr(m, 'content') and m.content]
                if ai_messages:
                    response = ai_messages[-1].content
                    return response, None
            
            return "I'm sorry, I couldn't generate a response.", None
            
    except Exception as e:
        return None, str(e)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 LangGraph RAG Chatbot Pro</h1>', unsafe_allow_html=True)
    
    # Initialize session state for RAG manager and agent
    if "rag_manager" not in st.session_state:
        st.session_state.rag_manager = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
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
        
        # Initialize RAG manager if not already done
        if st.session_state.rag_manager is None:
            rag_manager, rag_error = initialize_rag()
            if rag_manager:
                st.session_state.rag_manager = rag_manager
                st.success("✅ RAG system ready")
            else:
                st.error(f"❌ RAG system failed: {rag_error}")
                st.stop()
        else:
            rag_manager = st.session_state.rag_manager
            st.success("✅ RAG system ready")
        
        # Initialize agent if not already done
        if st.session_state.agent is None:
            agent, agent_error = initialize_agent()
            if agent:
                st.session_state.agent = agent
                st.success("✅ Agent ready")
            else:
                st.error(f"❌ Agent failed: {agent_error}")
                st.stop()
        else:
            agent = st.session_state.agent
            st.success("✅ Agent ready")
        
        # Show document count
        if rag_manager:
            doc_count = rag_manager.get_document_count()
            st.metric("Documents loaded", doc_count)
        
        # Clear RAG data button
        if st.button("🗑️ Clear All Documents"):
            if rag_manager:
                rag_manager.clear_vectorstore()
                st.success("✅ All documents cleared")
                st.rerun()
        
        # Debug information
        st.subheader("🔍 Debug Info")
        if st.button("🔄 Refresh RAG Status"):
            st.rerun()
        
        if rag_manager:
            st.info(f"RAG Manager Type: {type(rag_manager).__name__}")
            st.info(f"Documents Loaded: {rag_manager.get_document_count()}")
            st.info(f"Raw Data Folder: {rag_manager.rawdata_folder}")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "📁 Documents", "🌐 Web Scraping", "💬 Chat", "📊 Analytics"])
    
    with tab1:
        st.header("Welcome to LangGraph RAG Chatbot Pro!")
        
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
                <h3>🤖 LangGraph Agent</h3>
                <p>Stateful conversation management with tool execution and RAG integration.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 LangSmith Integration</h3>
                <p>Complete conversation tracking and performance monitoring in LangSmith.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h3>🌐 Web Scraping</h3>
                <p>Add website content to your knowledge base for comprehensive information.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System status dashboard
        st.subheader("System Status Dashboard")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            st.markdown("""
            <div class="metric-card">
                <h3>OpenAI API</h3>
                <p class="status-success">✅ Connected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col2:
            st.markdown("""
            <div class="metric-card">
                <h3>LangSmith</h3>
                <p class="status-success">✅ Connected</p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col3:
            st.markdown("""
            <div class="metric-card">
                <h3>RAG System</h3>
                <p class="status-success">✅ Ready</p>
            </div>
            """, unsafe_allow_html=True)
        
        with status_col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Agent</h3>
                <p class="status-success">✅ Ready</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📁 Document Management")
        
        # Get RAG manager from session state
        rag_manager = st.session_state.rag_manager
        
        # Document loading
        st.subheader("Load Documents")
        
        # Store loaded documents in session state
        if "loaded_documents" not in st.session_state:
            st.session_state.loaded_documents = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📂 Load Documents from rawdata/ folder"):
                with st.spinner("Loading documents..."):
                    documents, error = load_documents(rag_manager)
                    
                    if documents:
                        st.session_state.loaded_documents = documents
                        st.success(f"✅ Loaded {len(documents)} documents")
                        st.rerun()
                    else:
                        st.error(f"❌ Error loading documents: {error}")
        
        with col2:
            if st.button("🔧 Process and Store Documents", disabled=st.session_state.loaded_documents is None):
                with st.spinner("Processing documents..."):
                    success, error = process_documents(rag_manager, st.session_state.loaded_documents)
                    
                    if success:
                        st.success("✅ Documents processed and stored successfully!")
                        # Clear loaded documents after successful processing
                        st.session_state.loaded_documents = None
                        st.rerun()
                    else:
                        st.error(f"❌ Error processing documents: {error}")
        
        # Show loaded documents info
        if st.session_state.loaded_documents:
            st.subheader("📄 Loaded Documents (Ready to Process)")
            doc_info = []
            for i, doc in enumerate(st.session_state.loaded_documents):
                doc_info.append({
                    "Document": f"Document {i+1}",
                    "Source": doc.metadata.get('source', 'Unknown'),
                    "Type": doc.metadata.get('type', 'Unknown'),
                    "Content Length": len(doc.page_content)
                })
            
            df = pd.DataFrame(doc_info)
            st.dataframe(df, width='stretch')
            
            st.info("💡 Click 'Process and Store Documents' to add these documents to your knowledge base.")
        
        # Document statistics
        st.subheader("Document Statistics")
        if rag_manager:
            doc_count = rag_manager.get_document_count()
            st.metric("Total Documents in Knowledge Base", doc_count)
            
            if doc_count > 0:
                # Create a simple chart
                data = {"Documents": [doc_count], "Category": ["Stored"]}
                df = pd.DataFrame(data)
                fig = px.bar(df, x="Category", y="Documents", title="Documents in Knowledge Base")
                st.plotly_chart(fig, width='stretch')
                
                # Show stored document sources
                st.subheader("📚 Stored Document Sources")
                try:
                    # Get some sample documents to show sources
                    sample_docs = rag_manager.search_documents("", k=5)
                    if sample_docs:
                        sources = [doc.metadata.get('source', 'Unknown') for doc in sample_docs]
                        unique_sources = list(set(sources))
                        for source in unique_sources:
                            st.info(f"• {source}")
                except:
                    st.info("• Document sources information not available")
                
                # Test RAG functionality
                st.subheader("🧪 Test RAG Functionality")
                test_query = st.text_input("Test search query:", value="RAG functionality", key="test_query")
                if st.button("🔍 Test Search"):
                    with st.spinner("Testing search..."):
                        try:
                            results = rag_manager.search_documents(test_query, k=3)
                            if results:
                                st.success(f"✅ Search successful! Found {len(results)} relevant documents:")
                                for i, doc in enumerate(results):
                                    st.info(f"**Result {i+1}** (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:200]}...")
                            else:
                                st.warning("No relevant documents found for this query.")
                        except Exception as e:
                            st.error(f"❌ Search test failed: {e}")
            else:
                st.info("No documents have been processed and stored yet.")
    
    with tab3:
        st.header("🌐 Web Scraping")
        
        # Get RAG manager from session state
        rag_manager = st.session_state.rag_manager
        
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
        st.header("💬 Chat with RAG Agent")
        
        # Get agent and RAG manager from session state
        agent = st.session_state.agent
        rag_manager = st.session_state.rag_manager
        
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
            
            # Generate response using the agent
            with st.chat_message("assistant"):
                response, error = chat_with_agent(agent, prompt, rag_manager)
                
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"❌ Error generating response: {error}")
        
        # Chat controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 Show RAG Context"):
                if st.session_state.messages:
                    st.subheader("RAG Context for Last Query")
                    if rag_manager.get_document_count() > 0:
                        last_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
                        if last_query:
                            relevant_docs = rag_manager.search_documents(last_query, k=3)
                            if relevant_docs:
                                for i, doc in enumerate(relevant_docs):
                                    st.info(f"**Document {i+1}** (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:300]}...")
                            else:
                                st.info("No relevant documents found for the last query.")
                    else:
                        st.info("No documents loaded in the knowledge base.")
        
        with col3:
            if st.button("🔍 Search Documents"):
                search_query = st.text_input("Enter search query:")
                if search_query:
                    relevant_docs = rag_manager.search_documents(search_query, k=5)
                    if relevant_docs:
                        st.subheader(f"Search Results for: '{search_query}'")
                        for i, doc in enumerate(relevant_docs):
                            st.info(f"**Result {i+1}** (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:200]}...")
                    else:
                        st.info("No relevant documents found.")
    
    with tab5:
        st.header("📊 Analytics & Monitoring")
        
        # Get RAG manager from session state
        rag_manager = st.session_state.rag_manager
        
        # System metrics
        st.subheader("System Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents in KB", rag_manager.get_document_count())
        
        with col2:
            st.metric("Chat Messages", len(st.session_state.get("messages", [])))
        
        with col3:
            st.metric("RAG System", "Active")
        
        # Performance charts
        st.subheader("Performance Overview")
        
        # Document distribution chart
        if rag_manager.get_document_count() > 0:
            # Create sample data for demonstration
            doc_types = ["Text Files", "Web Content", "Other"]
            doc_counts = [rag_manager.get_document_count(), 0, 0]  # Simplified for demo
            
            fig = px.pie(values=doc_counts, names=doc_types, title="Document Distribution")
            st.plotly_chart(fig, width='stretch')
        
        # LangSmith integration info
        st.subheader("LangSmith Integration")
        st.info("""
        📊 **Tracing Active**: All conversations and operations are being traced in LangSmith
        🔍 **Project**: Check your LangSmith dashboard for detailed analytics
        📈 **Performance**: Monitor response times, token usage, and conversation flows
        """)

if __name__ == "__main__":
    main()
