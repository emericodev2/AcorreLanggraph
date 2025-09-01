from setuptools import setup, find_packages

setup(
    name="local-placeholder-dep",
    version="0.0.0",
    description="Placeholder dependency so LangGraph build step /deps/* succeeds",
    packages=find_packages(),
    python_requires=">=3.11",
)
