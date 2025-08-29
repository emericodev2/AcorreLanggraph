try:
    from app.agent import build_agent
except ImportError as e:
    print(f"Import error: {e}")
    raise

# LangGraph Cloud should call this to obtain the compiled graph at runtime
# so that required environment variables (e.g., OPENAI_API_KEY) are available.

def create_graph():
    try:
        return build_agent()
    except Exception as e:
        print(f"Error building agent: {e}")
        raise
