import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from pathlib import Path

from app.agent import build_agent
from app.rag import get_rag_manager
from langchain_core.messages import HumanMessage


def setup_rag() -> None:
	"""Setup RAG by loading documents and optionally scraping websites"""
	console = Console()
	rag_manager = get_rag_manager()
	
	console.print(Panel(
		"[bold blue]RAG Setup - Knowledge Base Initialization[/bold blue]\n\n"
		"This will help you set up your knowledge base with documents and websites.",
		title="🤖 RAG Setup",
		border_style="blue"
	))
	
	# Check if rawdata folder has documents
	rawdata_path = Path("rawdata")
	if rawdata_path.exists() and any(rawdata_path.iterdir()):
		console.print(f"📁 Found existing documents in '{rawdata_path}' folder")
		
		if Confirm.ask("Do you want to load documents from the rawdata folder?"):
			console.print("Loading documents...")
			documents = rag_manager.load_documents_from_folder()
			
			if documents:
				success = rag_manager.process_and_store_documents(documents)
				if success:
					console.print(f"✅ Successfully loaded {len(documents)} documents!")
				else:
					console.print("❌ Failed to process documents")
			else:
				console.print("⚠️  No documents found to load")
	else:
		console.print(f"📁 No documents found in '{rawdata_path}' folder")
		console.print("💡 Tip: Add .txt, .pdf, .docx, .md, or .csv files to the 'rawdata' folder")
	
	# Ask about website scraping
	if Confirm.ask("Do you want to scrape a website and add it to your knowledge base?"):
		while True:
			url = Prompt.ask("Enter website URL")
			if url.strip().lower() in ['skip', 'no', 'exit']:
				break
			
			console.print(f"🌐 Scraping {url}...")
			documents = rag_manager.scrape_website(url)
			
			if documents:
				success = rag_manager.process_and_store_documents(documents)
				if success:
					console.print(f"✅ Successfully scraped and stored {url}!")
				else:
					console.print("❌ Failed to store scraped content")
			else:
				console.print(f"❌ Failed to scrape {url}")
			
			if not Confirm.ask("Scrape another website?"):
				break
	
	# Show final stats
	doc_count = rag_manager.get_document_count()
	console.print(Panel(
		f"[green]Knowledge base setup complete![/green]\n\n"
		f"📊 Total document chunks: {doc_count}\n"
		f"💾 Vector database location: {rag_manager.persist_directory}",
		title="✅ Setup Complete",
		border_style="green"
	))


def main() -> None:
	# Load environment variables
	load_dotenv()
	
	console = Console()
	
	# Check required environment variables
	required_vars = {
		"OPENAI_API_KEY": "OpenAI API key for model access",
		"LANGCHAIN_API_KEY": "LangSmith API key for tracing",
		"LANGCHAIN_PROJECT": "LangSmith project name for organizing traces"
	}
	
	missing_vars = []
	for var, description in required_vars.items():
		if not os.getenv(var):
			missing_vars.append(f"• {var}: {description}")
	
	if missing_vars:
		console.print(Panel(
			f"[red]Missing required environment variables:[/red]\n\n" + "\n".join(missing_vars) + 
			"\n\nPlease check your .env file or set these environment variables.",
			title="Configuration Error",
			border_style="red"
		))
		return
	
	# Display configuration info
	console.print(Panel(
		f"[green]Configuration loaded successfully![/green]\n\n"
		f"• OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}\n"
		f"• LangSmith Project: {os.getenv('LANGCHAIN_PROJECT')}\n"
		f"• Tracing: {'Enabled' if os.getenv('LANGCHAIN_TRACING_V2') == 'true' else 'Disabled'}",
		title="LangGraph Agent + LangSmith Tracing + RAG",
		border_style="green"
	))
	
	# Ask about RAG setup
	if Confirm.ask("Do you want to set up RAG (Retrieval-Augmented Generation)?"):
		setup_rag()
	
	try:
		agent = build_agent()
		state = {"messages": []}

		console.print("\n[bold green]LangGraph Chatbot with RAG[/bold green] (type 'exit' to quit)")
		console.print("[dim]All conversations will be traced in LangSmith and enhanced with RAG[/dim]\n")
		
		# Show available commands
		console.print(Panel(
			"[bold]Available Commands:[/bold]\n"
			"• Type your question normally for RAG-enhanced responses\n"
			"• Use tools like 'load_documents', 'scrape_website', 'search_knowledge_base'\n"
			"• Type 'help' to see available tools\n"
			"• Type 'stats' to see knowledge base statistics",
			title="💡 Usage Tips",
			border_style="yellow"
		))
		
		while True:
			text = Prompt.ask("You")
			if text.strip().lower() in {"exit", "quit"}:
				break
			
			# Handle special commands
			if text.strip().lower() == "help":
				console.print(Panel(
					"[bold]Available Tools:[/bold]\n"
					"• load_documents - Load documents from rawdata folder\n"
					"• scrape_website - Scrape and add website content\n"
					"• search_knowledge_base - Search the knowledge base\n"
					"• get_knowledge_base_stats - Show database statistics\n"
					"• clear_knowledge_base - Clear all documents\n"
					"• get_time - Get current time\n"
					"• echo - Echo back your text",
					title="🛠️  Tools",
					border_style="cyan"
				))
				continue
			
			if text.strip().lower() == "stats":
				rag_manager = get_rag_manager()
				doc_count = rag_manager.get_document_count()
				console.print(f"📊 Knowledge base contains {doc_count} document chunks")
				continue
			
			state["messages"].append(HumanMessage(content=text))
			state = agent.invoke(state)
			
			# Find the last AIMessage in the state
			ai_messages = [m for m in state["messages"] if m.__class__.__name__ == "AIMessage"]
			if ai_messages:
				console.print(f"[bold cyan]Bot[/bold cyan]: {ai_messages[-1].content}")
		
		console.print("\n[green]Chat session ended. Check LangSmith dashboard for traces![/green]")
		
	except Exception as e:
		console.print(Panel(
			f"[red]Error initializing agent:[/red]\n{str(e)}",
			title="Initialization Error",
			border_style="red"
		))


if __name__ == "__main__":
	main()
