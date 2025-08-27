# MCP Protocol Server for Direct Claude Integration
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

try:
    from mcp import Server, ListToolsResult, CallToolResult, Tool
    from mcp.types import TextContent, ImageContent, EmbeddedResource
except ImportError:
    # Fallback implementation if MCP is not available
    class Server:
        def __init__(self, name: str, version: str): 
            self.name = name
            self.version = version
            self._tools = {}
            
        def list_tools(self): 
            def decorator(func): 
                self._list_tools_func = func
                return func
            return decorator
            
        def call_tool(self): 
            def decorator(func):
                self._call_tool_func = func
                return func
            return decorator
        
    class Tool:
        def __init__(self, name: str, description: str, inputSchema: dict):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text
            
    class ListToolsResult:
        def __init__(self, tools: List[Tool]):
            self.tools = tools
            
    class CallToolResult:
        def __init__(self, content: List[Union[TextContent, Any]]):
            self.content = content

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import EnhancedOrchestrator
from src.utils.config import Config
from src.utils.monitoring import get_metrics, track_mcp_call

logger = logging.getLogger(__name__)

class MCPMultiAgentServer:
    """MCP Protocol Server for Multi-Agent PostgreSQL Analysis"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.server = Server("multi-agent-postgres-analysis", "1.0.0")
        self.orchestrator = None
        self.session_id = str(uuid.uuid4())
        
        # Setup tools
        self._setup_tools()
        
    async def initialize(self):
        """Initialize the MCP server and orchestrator"""
        try:
            self.orchestrator = EnhancedOrchestrator(self.config)
            await self.orchestrator.initialize()
            logger.info(f"MCP Server initialized with session: {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            return False
    
    def _setup_tools(self):
        """Setup MCP tools for Claude integration"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools for Claude"""
            tools = [
                Tool(
                    name="analyze_databases",
                    description="Analyze data across multiple PostgreSQL databases with intelligent query planning and cross-database correlation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query to analyze data across databases"
                            },
                            "include_insights": {
                                "type": "boolean",
                                "description": "Whether to include AI-generated insights (default: true)",
                                "default": True
                            },
                            "max_records": {
                                "type": "integer",
                                "description": "Maximum number of records to return (default: 1000)",
                                "default": 1000
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_database_schema",
                    description="Get schema information for configured databases",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "database_name": {
                                "type": "string",
                                "description": "Specific database to analyze (optional, analyzes all if not specified)"
                            }
                        }
                    }
                ),
                Tool(
                    name="test_connections",
                    description="Test connectivity to all configured databases",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_details": {
                                "type": "boolean", 
                                "description": "Include detailed connection information (default: false)",
                                "default": False
                            }
                        }
                    }
                ),
                Tool(
                    name="get_system_metrics",
                    description="Get system performance metrics and query statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_history": {
                                "type": "boolean",
                                "description": "Include query history metrics (default: true)",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="explain_query_plan",
                    description="Explain how a query would be executed across the multi-agent system",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to explain"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
            """Handle tool calls from Claude"""
            
            start_time = datetime.now()
            
            try:
                if name == "analyze_databases":
                    result = await self._handle_analyze_databases(arguments)
                elif name == "get_database_schema":
                    result = await self._handle_get_schema(arguments)
                elif name == "test_connections":
                    result = await self._handle_test_connections(arguments)
                elif name == "get_system_metrics":
                    result = await self._handle_get_metrics(arguments)
                elif name == "explain_query_plan":
                    result = await self._handle_explain_plan(arguments)
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown tool: {name}",
                        "available_tools": ["analyze_databases", "get_database_schema", "test_connections", "get_system_metrics", "explain_query_plan"]
                    }
                
                # Track the call for monitoring
                execution_time = (datetime.now() - start_time).total_seconds()
                track_mcp_call(name, result.get("success", False), execution_time)
                
                # Format result for Claude
                content = [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                return CallToolResult(content=content)
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "error": str(e),
                    "tool": name,
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                content = [TextContent(type="text", text=json.dumps(error_result, indent=2))]
                return CallToolResult(content=content)
    
    async def _handle_analyze_databases(self, arguments: dict) -> dict:
        """Handle database analysis requests"""
        if not self.orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        query = arguments.get("query", "")
        include_insights = arguments.get("include_insights", True)
        max_records = arguments.get("max_records", 1000)
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        try:
            # Process through enhanced orchestrator
            result = await self.orchestrator.process_query(
                query=query,
                include_insights=include_insights,
                max_records=max_records,
                session_id=self.session_id
            )
            
            return {
                "success": True,
                "query": query,
                "results": result,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "session_id": self.session_id
            }
    
    async def _handle_get_schema(self, arguments: dict) -> dict:
        """Handle schema information requests"""
        if not self.orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        database_name = arguments.get("database_name")
        
        try:
            schema_info = await self.orchestrator.get_database_schemas(database_name)
            
            return {
                "success": True,
                "schema_info": schema_info,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "database_name": database_name,
                "session_id": self.session_id
            }
    
    async def _handle_test_connections(self, arguments: dict) -> dict:
        """Handle connection testing requests"""
        if not self.orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        include_details = arguments.get("include_details", False)
        
        try:
            connection_results = await self.orchestrator.test_all_connections(include_details)
            
            return {
                "success": True,
                "connection_results": connection_results,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id
            }
    
    async def _handle_get_metrics(self, arguments: dict) -> dict:
        """Handle system metrics requests"""
        include_history = arguments.get("include_history", True)
        
        try:
            metrics = get_metrics(include_history=include_history)
            
            return {
                "success": True,
                "metrics": metrics,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id
            }
    
    async def _handle_explain_plan(self, arguments: dict) -> dict:
        """Handle query plan explanation requests"""
        if not self.orchestrator:
            return {"success": False, "error": "System not initialized"}
        
        query = arguments.get("query", "")
        
        if not query:
            return {"success": False, "error": "Query is required"}
        
        try:
            execution_plan = await self.orchestrator.explain_query_plan(query)
            
            return {
                "success": True,
                "query": query,
                "execution_plan": execution_plan,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "session_id": self.session_id
            }
    
    async def cleanup(self):
        """Cleanup server resources"""
        if self.orchestrator:
            await self.orchestrator.cleanup()
        logger.info(f"MCP Server cleanup completed for session: {self.session_id}")

# Factory function for easy server creation
async def create_mcp_server(config: Optional[Config] = None) -> MCPMultiAgentServer:
    """Create and initialize MCP server"""
    server = MCPMultiAgentServer(config)
    await server.initialize()
    return server

# CLI entry point for MCP server
async def run_mcp_server():
    """Run MCP server standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent PostgreSQL Analysis MCP Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        server = await create_mcp_server()
        logger.info(f"MCP Server running on {args.host}:{args.port}")
        logger.info("Available tools: analyze_databases, get_database_schema, test_connections, get_system_metrics, explain_query_plan")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
        await server.cleanup()
    except Exception as e:
        logger.error(f"MCP server error: {e}")

if __name__ == "__main__":
    asyncio.run(run_mcp_server())