#!/bin/bash
echo "Starting Enhanced Multi-Agent System with NPM MCP Server"
echo "========================================================="
echo

# Start MCP Database Server using npm in background
echo "Starting MCP Database Server (npm package)..."
npm run mcp-postgres &
MCP_PID=$!

# Wait for MCP server to start
sleep 5

# Start Streamlit web interface
echo "Starting Streamlit Web Interface..."
echo
echo "Services will be available at:"
echo "- Web Interface: http://localhost:8501"
echo "- MCP Database Server: Available for Claude integration"
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