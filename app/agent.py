import os
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.tools import TOOLS
from app.rag import get_rag_manager


# Load environment variables
load_dotenv()

# Configure LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "AcorreLangchain")


def _make_tools() -> Dict[str, StructuredTool]:
	structured: Dict[str, StructuredTool] = {}
	for name, meta in TOOLS.items():
		structured[name] = StructuredTool.from_function(
			func=meta["func"],
			name=name,
			description=meta["description"],
			args_schema=None,
		)
	return structured


def build_agent() -> Any:
	# Verify OpenAI API key is set
	if not os.getenv("OPENAI_API_KEY"):
		raise ValueError("OPENAI_API_KEY environment variable is required")
	
	model = ChatOpenAI(
		model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
		temperature=0,
		openai_api_key=os.getenv("OPENAI_API_KEY")
	)
	
	tools = _make_tools()
	model_with_tools = model.bind_tools(list(tools.values()))

	def agent_node(state: GraphState) -> GraphState:
		messages: List[Any] = state.get("messages", [])
		if not messages:
			messages = [SystemMessage(content="You are a helpful assistant with access to a knowledge base. Use the available tools to help users and provide accurate information based on the documents and data available.")]
		
		# Get the latest user message
		latest_message = messages[-1] if messages else None
		
		# If this is a user message, try to enhance it with RAG context
		if isinstance(latest_message, HumanMessage) and latest_message.content:
			try:
				rag_manager = get_rag_manager()
				document_count = rag_manager.get_document_count()
				
				if document_count > 0:
					# Search for relevant documents
					relevant_docs = rag_manager.search_documents(latest_message.content, k=3)
					
					if relevant_docs:
						# Create context from relevant documents
						context = "\n\n".join([
							f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content[:500]}..."
							for i, doc in enumerate(relevant_docs)
						])
						
						# Enhance the system message with context
						enhanced_system = f"""You are a helpful assistant with access to a knowledge base containing {document_count} documents.

Relevant context for the current query:
{context}

Use this information to provide accurate and helpful responses. If the user's question relates to the documents, incorporate relevant information from them. If not, use your general knowledge to help the user.

Available tools: {', '.join(tools.keys())}"""
						
						# Update the first message (system message)
						if messages and isinstance(messages[0], SystemMessage):
							messages[0] = SystemMessage(content=enhanced_system)
						else:
							messages.insert(0, SystemMessage(content=enhanced_system))
			except Exception as e:
				# If RAG fails, continue without it
				print(f"RAG context enhancement failed: {e}")
		
		response = model_with_tools.invoke(messages)

		# If the model requested a tool, execute it and append ToolMessage
		if hasattr(response, "tool_calls") and response.tool_calls:
			for call in response.tool_calls:
				tool_name = call["name"]
				args = call.get("args", {})
				tool = tools.get(tool_name)
				if tool is None:
					continue
				result = tool.func(args)
				messages.append(ToolMessage(
					content=str(result), 
					tool_name=tool_name, 
					tool_call_id=call.get("id", tool_name)
				))
			# Re-ask model with tool results
			final = model_with_tools.invoke(messages + [response])
			messages.append(final)
			return {"messages": messages}

		messages.append(response)
		return {"messages": messages}

	# Updated for LangGraph 0.4.0+ compatibility
	workflow = StateGraph(GraphState)
	workflow.add_node("agent", agent_node)
	workflow.set_entry_point("agent")
	workflow.set_finish_point("agent")
	
	# Use the new compilation method
	return workflow.compile()
