#!/bin/bash
echo "Starting Distributed Multi-Agent System"
echo "======================================="
echo

echo "Setting up distributed databases..."
npm run create-demo-db
npm run setup-distributed-dbs

echo
echo "Starting distributed MCP servers and web interface..."
echo "- Users MCP Server: ./data/users_db.sqlite"
echo "- Products MCP Server: ./data/products_db.sqlite"
echo "- Sales MCP Server: ./data/sales_db.sqlite"  
echo "- Web Interface: http://localhost:8501"
echo

# Start all services with concurrently
npm run start-distributed