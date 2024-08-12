from setuptools import setup, find_packages

setup(
    name="simple_llm_tool",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple tool library to run LLM models with GGUF format",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simple_llm_tool",
)
