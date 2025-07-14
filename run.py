#!/usr/bin/env python3
"""
Voice-Synced Text Display System - FastAPI Version
Real-time speech recognition with synchronized text display
"""

import sys
import os
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'whisper',
        'sounddevice',
        'numpy',
        'pandas',
        'python-multipart'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        # Handle special cases for package names with hyphens
        if package == 'python-multipart':
            module_name = 'multipart'
        elif package == 'openai-whisper':
            module_name = 'whisper'
        else:
            module_name = package.replace('-', '_')
        
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   uv sync")
        print("   or")
        print("   pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎤 Voice-Synced Text Display System 🖥️")
    print("")
    print("Real-time speech recognition with synchronized display")
    print("                (FastAPI Version)")
    print("="*70)
    print("")
    
    print("🔧 Starting Voice-Synced Text Display System (FastAPI)...")
    print("⏳ Loading Whisper model (this may take a moment on first run)...")
    print("")
    
    print("🚀 Quick Start Guide:")
    print("")
    print("1. The system will start automatically")
    print("2. Open your web browser to:")
    print("   • Main Display: http://localhost:8000")
    print("   • Admin Panel: http://localhost:8000/admin")
    print("   • API Docs: http://localhost:8000/docs")
    print("")
    print("3. In the Admin Panel:")
    print("   • Upload your scripts and translations")
    print("   • Configure keywords")
    print("   • Start speech recognition")
    print("")
    print("4. On the Display:")
    print("   • Use fullscreen mode for projection")
    print("   • Press SPACEBAR to start/stop listening")
    print("   • Press R to reset line counter")
    print("")
    print("📋 Controls:")
    print("   • SPACEBAR: Start/Stop speech recognition")
    print("   • R: Reset line counter")
    print("   • Ctrl+C: Stop the server")
    print("")
    print("🔧 Troubleshooting:")
    print("   • Ensure microphone permissions are granted")
    print("   • Speak clearly and follow your script")
    print("   • Check admin panel for system status")
    print("   • View API docs at /docs for debugging")
    print("")
    
    print("🚀 Starting FastAPI server with hot reload...")
    print("📡 Server will be available at:")
    print("   • http://localhost:8000 (Display)")
    print("   • http://localhost:8000/admin (Admin Panel)")
    print("   • http://localhost:8000/docs (API Documentation)")
    print("")
    print("🔄 Hot reload enabled - changes to app.py will auto-restart")
    print("⏹️  Press Ctrl+C to stop the server")
    print("")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start the FastAPI server with hot reload
    try:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Enable hot reload
            reload_dirs=["."],  # Watch current directory for changes
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 