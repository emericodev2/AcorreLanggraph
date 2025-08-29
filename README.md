## LangGraph Agent + Chatbot with RAG (Python)

A powerful LangChain + LangGraph agent with RAG (Retrieval-Augmented Generation), tools, and **both CLI and Web UI interfaces**.

### Features
- **LangGraph Agent**: Stateful conversation management with tool execution
- **RAG Integration**: Document loading, web scraping, and semantic search
- **LangSmith Tracing**: Complete conversation and operation tracking
- **Dual Interface**: Beautiful web UI + powerful CLI
- **Multiple File Formats**: Support for TXT, PDF, DOCX, MD, CSV files
- **Web Scraping**: Add website content to your knowledge base

### Prerequisites
- Python 3.9+
- OpenAI API key
- LangSmith API key (for tracing)

### Setup
1. Create a virtualenv (PowerShell):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Configure environment:
```powershell
copy .env.example .env
# edit .env and set:
# OPENAI_API_KEY=your_openai_key
# LANGCHAIN_API_KEY=your_langsmith_key
# LANGCHAIN_PROJECT=your_project_name
```

4. Add documents to the `rawdata/` folder (optional):
   - Supported formats: TXT, PDF, DOCX, MD, CSV
   - The chatbot will automatically load and index these documents

### 🚀 Quick Start

#### Option 1: Interactive Startup (Recommended)
```powershell
.\.venv\Scripts\python start_web_app.py
```
This will let you choose between Web UI and CLI interfaces.

#### Option 2: Web UI (Beautiful Interface)
```powershell
.\.venv\Scripts\python -m streamlit run web_app_enhanced.py
```
- Opens in your browser at `http://localhost:8501`
- Beautiful, interactive interface with tabs
- Real-time document management
- Visual analytics and charts

#### Option 3: CLI (Command Line)
```powershell
.\.venv\Scripts\python cli.py
```
- Traditional command-line interface
- Full RAG functionality
- Interactive document setup

### 🌐 Web UI Features

The web interface provides:

#### 🏠 Home Tab
- System status dashboard
- Feature overview with cards
- Real-time metrics

#### 📁 Documents Tab
- Load documents from rawdata folder
- Process and store documents
- Document statistics and charts
- Visual document information

#### 🌐 Web Scraping Tab
- Add website content to knowledge base
- URL validation and processing
- Real-time scraping status

#### 💬 Chat Tab
- Full RAG-enhanced chat interface
- Chat history management
- RAG context visualization
- Document search integration

#### 📊 Analytics Tab
- System performance metrics
- Document distribution charts
- LangSmith integration status
- Real-time monitoring

### 💻 CLI Features

The command-line interface provides:
- **RAG Setup Wizard**: Guided document and website setup
- **Interactive Prompts**: Confirm actions and input URLs
- **Special Commands**: `help`, `stats`, and tool access
- **Rich Interface**: Beautiful panels and status updates

### RAG Setup
When you run either interface, it will guide you through RAG setup:

1. **Document Loading**: Automatically loads all documents from the `rawdata/` folder
2. **Web Scraping**: Optionally scrape websites and add them to your knowledge base
3. **Vector Database**: Creates a ChromaDB vector store for semantic search

### Usage
- **Normal Chat**: Ask questions and get RAG-enhanced responses
- **Document Management**: Load, process, and search through documents
- **Web Scraping**: Add website content to your knowledge base
- **Analytics**: Monitor system performance and document usage

### Files
- `web_app_enhanced.py`: **Main web application** with full RAG integration
- `web_app.py`: Basic web interface
- `start_web_app.py`: **Interactive startup script** (recommended)
- `cli.py`: Enhanced CLI with RAG setup and management
- `app/state.py`: Conversation state for LangGraph
- `app/tools.py`: Tools including RAG operations
- `app/agent.py`: LangGraph graph and agent node with RAG integration
- `app/rag.py`: RAG manager for document processing and retrieval
- `app/rag_fallback.py`: Simple RAG fallback system
- `rawdata/`: Folder for your documents (auto-created)
- `chroma_db/`: Vector database storage (auto-created)

### How RAG Works
1. **Document Ingestion**: Documents are loaded, chunked, and embedded
2. **Vector Storage**: Chunks are stored in ChromaDB with semantic embeddings
3. **Query Processing**: User questions trigger semantic search through stored documents
4. **Context Enhancement**: Relevant document chunks are retrieved and added to the conversation context
5. **Enhanced Responses**: The AI generates responses using both its knowledge and retrieved document context

### Supported Document Types
- **Text Files (.txt)**: Plain text documents
- **Markdown (.md)**: Markdown formatted documents
- **PDF (.pdf)**: PDF documents using PDFMiner
- **Word (.docx)**: Microsoft Word documents
- **CSV (.csv)**: Comma-separated value files

### Web Scraping
The chatbot can scrape websites and add their content to your knowledge base:
- Automatically handles URL validation
- Removes HTML tags and scripts
- Cleans and processes text content
- Stores with source metadata

### LangSmith Integration
All operations are automatically traced in LangSmith:
- Conversation history
- Tool executions
- RAG operations
- Performance metrics
- Debugging information

### 🎯 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure environment**: Set up your `.env` file
3. **Run startup script**: `python start_web_app.py`
4. **Choose interface**: Web UI (recommended) or CLI
5. **Set up RAG**: Load documents and/or scrape websites
6. **Start chatting**: Enjoy RAG-enhanced conversations!

### 💡 Tips for Best Experience

- **Web UI**: Best for document management and visual analytics
- **CLI**: Best for quick interactions and automation
- **Documents**: Start with a few text files to test the system
- **Web Scraping**: Use for adding external knowledge sources
- **LangSmith**: Monitor your conversations and system performance

The chatbot now provides both beautiful web interfaces and powerful command-line tools, making it easy to use regardless of your preference! 🎉
"# AcorreLanggraph" 
"# AcorreLanggraph" 
