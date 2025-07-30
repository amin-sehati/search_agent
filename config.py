import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the research system"""

    # API Keys - Set these as environment variables or modify directly
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # LangChain Configuration
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

    # Research Configuration
    MAX_SOURCES = 20
    DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
    FALLBACK_MODEL = "gpt-4.1-nano-2025-04-14"

    # Search Configuration
    SEARCH_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3

    # Report Configuration
    DEFAULT_REPORT_TYPE = "research_report"
    SUPPORTED_FORMATS = ["markdown", "html", "json"]

    # Tavily Configuration
    TAVILY_CONFIG = {
        "search_depth": "advanced",
        "include_answer": True,
        "include_raw_content": True,
        "max_results": 10,
    }

    # Firecrawl Configuration
    FIRECRAWL_CONFIG = {
        "page_options": {"onlyMainContent": True, "includeHtml": False},
        "limit": 10,
    }

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []

        if not cls.TAVILY_API_KEY:
            issues.append("Tavily API key not configured")

        if not cls.FIRECRAWL_API_KEY:
            issues.append("Firecrawl API key not configured")

        if not cls.OPENAI_API_KEY:
            issues.append("OpenAI API key not configured")

        # LangChain keys are optional for tracing
        if cls.LANGCHAIN_TRACING_V2 == "true" and not cls.LANGCHAIN_API_KEY:
            issues.append("LangChain API key required when tracing is enabled")

        return {"valid": len(issues) == 0, "issues": issues}

    @classmethod
    def get_example_queries(cls) -> Dict[str, str]:
        """Return example research queries for testing"""
        return {
            "technology": "latest developments in artificial intelligence 2024",
            "business": "remote work trends and productivity statistics",
            "science": "recent breakthroughs in quantum computing research",
            "health": "benefits and risks of intermittent fasting studies",
            "environment": "climate change impact on global agriculture",
        }
