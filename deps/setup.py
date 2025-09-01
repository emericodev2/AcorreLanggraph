from setuptools import setup, find_packages

setup(
    name="langgraph-deps",
    version="0.0.0",
    description="Dependencies for LangGraph deployment",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.3.0",
        "langgraph>=0.4.0",
        "langchain-openai>=0.3.0",
        "python-dotenv>=1.0.0",
        "langchain-community>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "chromadb>=0.4.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "rich>=13.7.1",
        "typer>=0.12.3",
        "streamlit>=1.32.0",
        "streamlit-chat>=0.1.1",
        "plotly>=5.18.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
)
