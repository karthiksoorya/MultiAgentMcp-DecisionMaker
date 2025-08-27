@echo off
echo Starting Enhanced Multi-Agent System with Demo SQLite Database
echo ==============================================================
echo.

REM Create demo database and setup data
echo Setting up demo SQLite database...
npm run create-demo-db

REM Initialize demo database with sample data if it doesn't exist
if not exist "data\demo.db" (
    echo Creating demo database with sample data...
    sqlite3 data\demo.db < data\setup_demo.sql 2>nul || echo SQLite not found, database will be created automatically by MCP server
)

REM Start MCP Database Server with SQLite in background
echo Starting MCP Database Server (SQLite)...
start "MCP Server" cmd /k "npm run mcp-sqlite"

REM Wait for MCP server to start
timeout /t 5 /nobreak > nul

REM Start Streamlit web interface
echo Starting Streamlit Web Interface...
echo.
echo Demo System Running:
echo - Web Interface: http://localhost:8501
echo - MCP Database Server: SQLite demo database
echo - Available Tools: read_query, write_query, create_table, list_tables, describe_table, export_query, append_insight, list_insights
echo.
echo Press Ctrl+C to stop Streamlit (MCP server will continue in separate window)
echo.
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost

pause