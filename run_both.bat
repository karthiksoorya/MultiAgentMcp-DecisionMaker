@echo off
echo Starting Multi-Agent System - Web Interface + MCP Server
echo ========================================================
echo.

REM Start MCP database server in background using npm package
echo Starting MCP Database Server (npm package)...
start "MCP Server" cmd /k "npx @executeautomation/database-server --postgresql --host localhost --database testdb --user testuser --password testpass --port 5432"

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