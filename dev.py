#!/usr/bin/env python3
"""
Development server with hot reload for Voice-Synced Text Display System
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting development server with hot reload...")
    print("📡 Server: http://localhost:8000")
    print("🔄 Hot reload enabled - changes will auto-restart")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        log_level="info"
    ) 