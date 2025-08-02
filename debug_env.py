#!/usr/bin/env python3
"""
Debug script to check environment variables in Vercel
"""
import os
import sys

def debug_environment():
    """Debug environment variables and Python environment"""
    
    print("=== ENVIRONMENT DEBUG INFO ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check for environment variables
    env_vars = [
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY", 
        "OPENAI_API_KEY",
        "VERCEL",
        "VERCEL_ENV",
        "VERCEL_REGION",
    ]
    
    print("\n=== ENVIRONMENT VARIABLES ===")
    for var in env_vars:
        value = os.environ.get(var) or os.getenv(var)
        if value:
            if "API_KEY" in var or "KEY" in var:
                # Mask sensitive values
                masked = f"{value[:8]}..." if len(value) > 8 else "****"
                print(f"{var}: {masked} (length: {len(value)})")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: NOT SET")
    
    # Check Python path and imports
    print(f"\n=== PYTHON PATH ===")
    for path in sys.path[:5]:  # Show first 5 entries
        print(f"  {path}")
    
    # Test imports
    print(f"\n=== IMPORT TESTS ===")
    try:
        import aiohttp
        print(f"✅ aiohttp: {aiohttp.__version__}")
    except ImportError as e:
        print(f"❌ aiohttp: {e}")
    
    try:
        import openai
        print(f"✅ openai: {openai.__version__}")
    except ImportError as e:
        print(f"❌ openai: {e}")
        
    try:
        from config import Config
        print(f"✅ config: imported successfully")
        
        # Test config validation
        validation = Config.validate_config()
        print(f"   Config validation: {validation}")
        
    except ImportError as e:
        print(f"❌ config: {e}")
    except Exception as e:
        print(f"❌ config validation: {e}")

if __name__ == "__main__":
    debug_environment()