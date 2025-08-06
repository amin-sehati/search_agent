#!/usr/bin/env python3
"""
Development server runner for AI Research Assistant
"""
import subprocess
import sys
import os


def main():
    print("🚀 Starting AI Research Assistant Development Server...")
    print("🌐 Next.js Application: http://localhost:3000")
    print("Press Ctrl+C to stop the server\n")

    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(__file__))

    try:
        # Start Next.js development server
        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        sys.exit(0)
    except FileNotFoundError:
        print("❌ Error: npm not found. Please install Node.js and npm.")
        print("💡 Try: https://nodejs.org/")
        sys.exit(1)


if __name__ == "__main__":
    main()
