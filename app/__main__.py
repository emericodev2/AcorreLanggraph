"""
Main entry point for the app package.
This allows running the app directly with: python -m app
"""

if __name__ == "__main__":
    from .agent import build_agent
    
    try:
        graph = build_agent()
        print("✅ Agent built successfully!")
        print(f"Graph type: {type(graph)}")
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        raise
