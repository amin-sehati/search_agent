#!/usr/bin/env python3
"""
Development server runner for AI Research Assistant
"""
import subprocess
import sys
import os
import time
import signal
from threading import Thread

def run_fastapi():
    """Run the FastAPI backend server"""
    os.chdir(os.path.join(os.path.dirname(__file__), 'api'))
    subprocess.run([sys.executable, 'main.py'])

def run_nextjs():
    """Run the Next.js frontend server"""
    os.chdir(os.path.dirname(__file__))
    subprocess.run(['npm', 'run', 'dev'])

def main():
    print("🚀 Starting AI Research Assistant Development Servers...")
    print("📊 FastAPI Backend: http://localhost:8000")
    print("🌐 Next.js Frontend: http://localhost:3000")
    print("Press Ctrl+C to stop both servers\n")
    
    # Start FastAPI in a separate thread
    fastapi_thread = Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Give FastAPI time to start
    time.sleep(2)
    
    try:
        # Start Next.js in main thread
        run_nextjs()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        sys.exit(0)

if __name__ == "__main__":
    main()