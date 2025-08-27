@echo off
echo Starting Multi-Agent System - Web Interface + MCP Server
echo ========================================================
echo.

REM Start MCP server in background
echo Starting MCP Server on port 8000...
start "MCP Server" cmd /k "python src/mcp/server.py --port 8000"

REM Wait a moment for MCP server to start
timeout /t 3 /nobreak > nul

REM Start Streamlit web interface
echo Starting Streamlit Web Interface on port 8501...
echo.
echo Services will be available at:
echo - Web Interface: http://localhost:8501
echo - MCP Server: http://localhost:8000
echo.
echo Press Ctrl+C to stop Streamlit (MCP server will continue in separate window)
echo.
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost

pause