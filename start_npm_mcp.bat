@echo off
echo Starting Enhanced Multi-Agent System with NPM MCP Server
echo ==========================================================
echo.

REM Start MCP Database Server using npm in background
echo Starting MCP Database Server (npm package)...
start "MCP Server" cmd /k "npm run mcp-postgres"

REM Wait for MCP server to start
timeout /t 5 /nobreak > nul

REM Start Streamlit web interface
echo Starting Streamlit Web Interface...
echo.
echo Services will be available at:
echo - Web Interface: http://localhost:8501
echo - MCP Database Server: Available for Claude integration
echo.
echo Press Ctrl+C to stop Streamlit (MCP server will continue in separate window)
echo.
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost

pause