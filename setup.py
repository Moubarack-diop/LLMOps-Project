from setuptools import find_packages, setup

setup(
    name="medassist",
    version="1.0.0",
    description="MedAssist — Assistant Médical Intelligent basé sur RAG",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mouhamed Diop",
    license="MIT",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=[
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-anthropic>=0.2.0",
        "langchain-huggingface>=0.0.3",
        "sentence-transformers>=2.7.0",
        "qdrant-client>=1.9.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "ragas>=0.1.9",
        "mlflow>=2.13.0",
        "datasets>=2.19.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=5.0.0",
            "httpx>=0.27.0",
            "flake8>=7.0.0",
            "black>=24.4.0",
        ]
    },
    entry_points={"console_scripts": ["medassist-api=src.api.main:app"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
