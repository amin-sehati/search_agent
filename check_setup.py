#!/usr/bin/env python3
"""
Setup verification script for AI Research Assistant
"""
import requests
import sys
import os
from config import Config

def check_config():
    """Check if configuration is valid"""
    print("🔧 Checking configuration...")
    try:
        config_status = Config.validate_config()
        if config_status["valid"]:
            print("✅ Configuration is valid")
            return True
        else:
            print("❌ Configuration issues:")
            for issue in config_status["issues"]:
                print(f"   • {issue}")
            return False
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_fastapi():
    """Check if FastAPI server is running"""
    print("\n🔌 Checking FastAPI server...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running on http://localhost:8000")
            return True
        else:
            print(f"❌ FastAPI server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ FastAPI server is not running on http://localhost:8000")
        print("   Run: cd api && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error checking FastAPI: {e}")
        return False

def check_nextjs():
    """Check if Next.js server is running"""
    print("\n🌐 Checking Next.js server...")
    try:
        # Try both default port and common alternative
        for port in [3000, 3001]:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Next.js server is running on http://localhost:{port}")
                    return True
            except requests.exceptions.ConnectionError:
                continue
        
        print("❌ Next.js server is not running on http://localhost:3000 or :3001")
        print("   Run: npm run dev")
        return False
    except Exception as e:
        print(f"❌ Error checking Next.js: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    # Check Python dependencies
    python_deps = ['fastapi', 'uvicorn', 'openai', 'aiohttp', 'langgraph', 'langchain']
    missing_python = []
    
    for dep in python_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_python.append(dep)
    
    if missing_python:
        print(f"❌ Missing Python dependencies: {', '.join(missing_python)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ Python dependencies are installed")
    
    # Check if Node.js dependencies are installed
    if not os.path.exists("node_modules"):
        print("❌ Node.js dependencies not installed")
        print("   Run: npm install")
        return False
    else:
        print("✅ Node.js dependencies are installed")
    
    return True

def main():
    print("🚀 AI Research Assistant Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
        ("FastAPI Backend", check_fastapi),
        ("Next.js Frontend", check_nextjs)
    ]
    
    results = []
    for name, check_func in checks:
        results.append(check_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All checks passed! Your setup is ready.")
        print("📖 Open http://localhost:3000 (or :3001) to use the application")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()