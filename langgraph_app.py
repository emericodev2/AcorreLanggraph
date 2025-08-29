from app.agent import build_agent

# LangGraph Cloud should call this to obtain the compiled graph at runtime
# so that required environment variables (e.g., OPENAI_API_KEY) are available.

def create_graph():
	return build_agent()
