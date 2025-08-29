from typing import Dict, Any
from datetime import datetime
from .rag import get_rag_manager


def get_time(_: Dict[str, Any]) -> str:
	"""Return the current time as an ISO string."""
	return datetime.now().isoformat(timespec="seconds")


def echo(input_data: Dict[str, Any]) -> str:
	"""Echo back the 'text' field from the input."""
	text = input_data.get("text")
	return f"You said: {text}" if text else "No text provided."


def load_documents(_: Dict[str, Any]) -> str:
	"""Load all documents from the rawdata folder and store them in the vector database."""
	try:
		rag_manager = get_rag_manager()
		documents = rag_manager.load_documents_from_folder()
		
		if documents:
			success = rag_manager.process_and_store_documents(documents)
			if success:
				return f"Successfully loaded and processed {len(documents)} documents. Vector database now contains {rag_manager.get_document_count()} chunks."
			else:
				return "Failed to process and store documents."
		else:
			return "No documents found in the rawdata folder. Please add some documents first."
	except Exception as e:
		return f"Error loading documents: {str(e)}"


def scrape_website(input_data: Dict[str, Any]) -> str:
	"""Scrape content from a website and add it to the knowledge base."""
	url = input_data.get("url")
	if not url:
		return "Please provide a 'url' parameter."
	
	try:
		rag_manager = get_rag_manager()
		documents = rag_manager.scrape_website(url)
		
		if documents:
			success = rag_manager.process_and_store_documents(documents)
			if success:
				return f"Successfully scraped and stored website content from {url}. Vector database now contains {rag_manager.get_document_count()} chunks."
			else:
				return "Failed to store scraped website content."
		else:
			return f"Failed to scrape content from {url}."
	except Exception as e:
		return f"Error scraping website: {str(e)}"


def search_knowledge_base(input_data: Dict[str, Any]) -> str:
	"""Search the knowledge base for relevant information."""
	query = input_data.get("query")
	if not query:
		return "Please provide a 'query' parameter."
	
	try:
		rag_manager = get_rag_manager()
		results = rag_manager.search_documents(query, k=3)
		
		if results:
			response = f"Found {len(results)} relevant documents for query: '{query}'\n\n"
			for i, doc in enumerate(results, 1):
				source = doc.metadata.get('source', 'Unknown')
				content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
				response += f"Document {i} (Source: {source}):\n{content}\n\n"
			return response
		else:
			return f"No relevant documents found for query: '{query}'"
	except Exception as e:
		return f"Error searching knowledge base: {str(e)}"


def get_knowledge_base_stats(_: Dict[str, Any]) -> str:
	"""Get statistics about the knowledge base."""
	try:
		rag_manager = get_rag_manager()
		document_count = rag_manager.get_document_count()
		return f"Knowledge base contains {document_count} document chunks."
	except Exception as e:
		return f"Error getting knowledge base stats: {str(e)}"


def clear_knowledge_base(_: Dict[str, Any]) -> str:
	"""Clear all documents from the knowledge base."""
	try:
		rag_manager = get_rag_manager()
		rag_manager.clear_vectorstore()
		return "Knowledge base has been cleared."
	except Exception as e:
		return f"Error clearing knowledge base: {str(e)}"


TOOLS = {
	"get_time": {
		"func": get_time,
		"description": "Get the current local time in ISO format.",
		"schema": {"type": "object", "properties": {}},
	},
	"echo": {
		"func": echo,
		"description": "Echo back user-provided text.",
		"schema": {
			"type": "object",
			"properties": {"text": {"type": "string"}},
			"required": ["text"],
		},
	},
	"load_documents": {
		"func": load_documents,
		"description": "Load all documents from the rawdata folder and store them in the vector database for RAG.",
		"schema": {"type": "object", "properties": {}},
	},
	"scrape_website": {
		"func": scrape_website,
		"description": "Scrape content from a website URL and add it to the knowledge base.",
		"schema": {
			"type": "object",
			"properties": {"url": {"type": "string"}},
			"required": ["url"],
		},
	},
	"search_knowledge_base": {
		"func": search_knowledge_base,
		"description": "Search the knowledge base for relevant information based on a query.",
		"schema": {
			"type": "object",
			"properties": {"query": {"type": "string"}},
			"required": ["query"],
		},
	},
	"get_knowledge_base_stats": {
		"func": get_knowledge_base_stats,
		"description": "Get statistics about the knowledge base including document count.",
		"schema": {"type": "object", "properties": {}},
	},
	"clear_knowledge_base": {
		"func": clear_knowledge_base,
		"description": "Clear all documents from the knowledge base.",
		"schema": {"type": "object", "properties": {}},
	},
}
