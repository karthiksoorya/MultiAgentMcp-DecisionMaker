#!/usr/bin/env python3
"""
Launch script to run both Web Interface and MCP Server simultaneously
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

class ServiceManager:
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        
    def start_mcp_server(self, port=8000):
        """Start MCP database server using npm package"""
        print(f"[MCP] Starting MCP Database Server (npm package)...")
        cmd = [
            "npx", "@executeautomation/database-server",
            "--postgresql", "--host", "localhost", "--database", "testdb",
            "--user", "testuser", "--password", "testpass", "--port", "5432"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(("MCP Server", process))
        return process
    
    def start_streamlit(self, port=8501):
        """Start Streamlit web interface"""
        print(f"[WEB] Starting Streamlit Web Interface on port {port}...")
        cmd = [
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", str(port),
            "--server.address", "localhost"
        ]
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        self.processes.append(("Streamlit", process))
        return process
    
    def cleanup(self):
        """Terminate all processes"""
        print("\n[STOP] Shutting down services...")
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"[OK] {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"[WARN] {name} force killed")
            except Exception as e:
                print(f"[ERROR] Error stopping {name}: {e}")
    
    def run(self):
        """Run both services"""
        try:
            # Start MCP server
            mcp_process = self.start_mcp_server(8000)
            
            # Give MCP server time to start
            print("[INIT] Waiting for MCP server to initialize...")
            time.sleep(3)
            
            # Check if MCP server started successfully
            if mcp_process.poll() is not None:
                stdout, stderr = mcp_process.communicate()
                print(f"[ERROR] MCP Server failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return
            
            # Start Streamlit
            streamlit_process = self.start_streamlit(8501)
            
            print("\n[SUCCESS] Services started successfully!")
            print("=" * 50)
            print("Web Interface: http://localhost:8501")
            print("MCP Server: http://localhost:8000")
            print("=" * 50)
            print("\n[MONITOR] Monitoring output (Press Ctrl+C to stop)...")
            
            # Monitor both processes
            while True:
                # Check if any process died
                for name, process in self.processes:
                    if process.poll() is not None:
                        print(f"\n[WARN] {name} stopped unexpectedly")
                        self.cleanup()
                        return
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[STOP] Received stop signal")
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    print("Multi-Agent PostgreSQL Analysis System")
    print("=====================================")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("[ERROR] streamlit_app.py not found. Please run from project root.")
        sys.exit(1)
    
    if not Path("src/mcp/server.py").exists():
        print("[ERROR] MCP server not found. Please run from project root.")
        sys.exit(1)
    
    # Start services
    manager = ServiceManager()
    manager.run()

if __name__ == "__main__":
    main()