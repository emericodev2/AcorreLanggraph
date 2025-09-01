"""
Main entry point for LangGraph Cloud deployment.
This file should be kept minimal and focused on graph creation.
"""

def create_graph():
    """
    Create and return the compiled LangGraph workflow.
    This function is called by LangGraph Cloud at runtime.
    """
    try:
        # Import here to ensure environment variables are available
        from app.agent import build_agent
        
        # Build and return the compiled graph
        graph = build_agent()
        return graph
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required dependencies are installed.")
        raise
    except Exception as e:
        print(f"Error building agent: {e}")
        print("Please check your configuration and dependencies.")
        raise
