#!/usr/bin/env python3
"""
Test script to verify the Multi-Agent system environment and components
"""
import os
import sys
from pathlib import Path

def test_environment():
    """Test the environment setup"""
    print("Testing Multi-Agent PostgreSQL Analysis System")
    print("=" * 60)
    
    # Test Python version
    print(f"[OK] Python version: {sys.version}")
    
    # Test working directory
    cwd = Path.cwd()
    print(f"[OK] Working directory: {cwd}")
    
    # Test environment file
    env_file = cwd / ".env"
    env_template = cwd / ".env.template"
    
    if env_file.exists():
        print("[OK] .env file exists")
    elif env_template.exists():
        print("[WARN] .env.template exists but .env file not found")
        print("       Please copy .env.template to .env and configure your settings")
    else:
        print("[ERROR] No environment configuration files found")
    
    # Test imports
    print("\nTesting component imports:")
    
    try:
        from src.utils.config import Config
        print("[OK] Configuration module")
    except ImportError as e:
        print(f"[ERROR] Configuration module: {e}")
        return False
    
    try:
        from src.agents.orchestrator import EnhancedOrchestrator
        print("[OK] Enhanced Orchestrator")
    except ImportError as e:
        print(f"[ERROR] Enhanced Orchestrator: {e}")
        return False
        
    try:
        from src.mcp.server import MCPMultiAgentServer
        print("[OK] MCP Server")
    except ImportError as e:
        print(f"[ERROR] MCP Server: {e}")
        return False
        
    try:
        import streamlit
        print(f"[OK] Streamlit (version: {streamlit.__version__})")
    except ImportError as e:
        print(f"[ERROR] Streamlit: {e}")
        return False
    
    # Test configuration loading
    print("\nTesting configuration:")
    try:
        config = Config()
        print("[OK] Configuration loaded successfully")
        
        # Test database configurations (without actual connection)
        if hasattr(config, 'databases') and config.databases:
            print(f"[OK] Found {len(config.databases)} database configurations")
        else:
            print("[WARN] No database configurations found (check .env file)")
            
    except Exception as e:
        print(f"[ERROR] Configuration loading failed: {e}")
        return False
    
    print("\nSystem Status Summary:")
    print("[OK] All core components imported successfully")
    print("[OK] System is ready for deployment")
    
    print("\nReady to launch!")
    print("Run: streamlit run streamlit_app.py")
    print("Or:  python src/mcp/server.py --port 8000")
    
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)