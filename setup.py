from setuptools import setup, find_packages

setup(
    name="acorre-langgraph",
    version="0.1.0",
    description="A LangGraph-based RAG agent for document processing and knowledge management",
    author="Your Name",
    author_email="your.email@example.com",
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
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
