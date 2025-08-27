#!/bin/bash
echo "Starting Multi-Agent System - Web Interface + MCP Server"
echo "========================================================"
echo

# Start MCP server in background
echo "Starting MCP Server on port 8000..."
python src/mcp/server.py --port 8000 &
MCP_PID=$!

# Wait a moment for MCP server to start
sleep 3

# Start Streamlit web interface
echo "Starting Streamlit Web Interface on port 8501..."
echo
echo "Services will be available at:"
echo "- Web Interface: http://localhost:8501"
echo "- MCP Server: http://localhost:8000"
echo
echo "Press Ctrl+C to stop both services"
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo "Shutting down services..."
    kill $MCP_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup INT TERM

# Start Streamlit (this will run in foreground)
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost

# If streamlit exits, cleanup MCP server
cleanup