# MCP Client for database operations
import asyncpg
from typing import Dict, Any, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class MCPClient:
    """Client for database operations via direct connection"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
    
    async def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = await asyncpg.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                user=self.db_config.username,
                password=self.db_config.password,
                database=self.db_config.database,
                ssl='require' if self.db_config.ssl else 'prefer'
            )
            logger.info(f"Connected to database: {self.db_config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.db_config.name}: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info(f"Disconnected from database: {self.db_config.name}")
    
    async def execute_query(self, sql_query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        if not self.connection:
            if not await self.connect():
                return []
        
        try:
            if params:
                rows = await self.connection.fetch(sql_query, *params)
            else:
                rows = await self.connection.fetch(sql_query)
            
            # Convert to list of dictionaries
            result = []
            for row in rows:
                result.append(dict(row))
            
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    async def list_tables(self) -> List[str]:
        """List all tables in the database"""
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        
        result = await self.execute_query(query)
        return [row["table_name"] for row in result]
    
    async def describe_table(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns 
        WHERE table_name = $1 
        AND table_schema = 'public'
        ORDER BY ordinal_position
        """
        
        result = await self.execute_query(query, [table_name])
        
        return {
            "table_name": table_name,
            "columns": [
                {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "nullable": row["is_nullable"] == "YES",
                    "default": row["column_default"],
                    "max_length": row["character_maximum_length"]
                }
                for row in result
            ]
        }
    
    async def get_sample_data(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from a table"""
        query = f"SELECT * FROM {table_name} LIMIT $1"
        return await self.execute_query(query, [limit])

class DatabaseAgent:
    """Database-specific agent that uses MCP client"""
    
    def __init__(self, agent_name: str, db_config, llm_client=None):
        self.agent_name = agent_name
        self.db_config = db_config
        self.mcp_client = MCPClient(db_config)
        self.llm_client = llm_client
    
    async def initialize(self) -> bool:
        """Initialize the agent and establish connections"""
        return await self.mcp_client.connect()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.mcp_client.disconnect()
    
    async def analyze_query_capability(self, user_query: str) -> Dict[str, Any]:
        """Analyze what this agent can provide for the user query"""
        
        # Get available tables
        tables = await self.mcp_client.list_tables()
        
        # Basic capability analysis
        capability_info = {
            "agent_name": self.agent_name,
            "database": self.db_config.name,
            "available_tables": tables,
            "can_help": len(tables) > 0,
            "capabilities": [],
            "requirements": []
        }
        
        # Add specific capabilities based on database type
        if "users" in self.agent_name.lower():
            capability_info["capabilities"] = [
                "User profiles and demographics",
                "User account information", 
                "User activity status"
            ]
            capability_info["requirements"] = ["None (primary data source)"]
            
        elif "transaction" in self.agent_name.lower():
            capability_info["capabilities"] = [
                "Transaction history",
                "Payment patterns",
                "Purchase behavior analysis"
            ]
            capability_info["requirements"] = ["user_ids from users database"]
        
        return capability_info
    
    async def execute_analysis(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute analysis based on user query and context"""
        
        capability = await self.analyze_query_capability(user_query)
        
        if not capability["can_help"]:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": "No tables available in this database",
                "data": []
            }
        
        # Execute appropriate queries based on agent type and context
        if "users" in self.agent_name.lower():
            return await self._handle_users_analysis(user_query, context)
        elif "transaction" in self.agent_name.lower():
            return await self._handle_transactions_analysis(user_query, context)
        else:
            return await self._handle_generic_analysis(user_query, context)
    
    async def _handle_users_analysis(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle user-related analysis"""
        
        # Basic user query - adjust this SQL to match your actual table schema
        query = """
        SELECT user_id, username, email, created_at, last_login, status
        FROM users 
        WHERE status = 'active'
        ORDER BY created_at DESC
        LIMIT 100
        """
        
        try:
            data = await self.mcp_client.execute_query(query)
            
            return {
                "agent_name": self.agent_name,
                "success": True,
                "data": data,
                "metadata": {
                    "record_count": len(data),
                    "query_type": "users_analysis",
                    "user_ids": [row["user_id"] for row in data]  # For other agents
                }
            }
            
        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def _handle_transactions_analysis(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle transaction-related analysis"""
        
        # Check if we have user_ids from context
        user_ids = []
        if context and "user_ids" in context:
            user_ids = context["user_ids"][:20]  # Limit for performance
        
        if user_ids:
            # Query for specific users - adjust SQL to match your schema
            placeholders = ",".join([f"${i+1}" for i in range(len(user_ids))])
            query = f"""
            SELECT user_id, transaction_date, amount, category, merchant
            FROM transactions 
            WHERE user_id IN ({placeholders})
            ORDER BY transaction_date DESC
            LIMIT 200
            """
            params = user_ids
        else:
            # General transaction query
            query = """
            SELECT user_id, transaction_date, amount, category, merchant
            FROM transactions 
            ORDER BY transaction_date DESC
            LIMIT 100
            """
            params = None
        
        try:
            data = await self.mcp_client.execute_query(query, params)
            
            return {
                "agent_name": self.agent_name,
                "success": True,
                "data": data,
                "metadata": {
                    "record_count": len(data),
                    "query_type": "transactions_analysis",
                    "used_context": bool(user_ids),
                    "context_user_count": len(user_ids) if user_ids else 0
                }
            }
            
        except Exception as e:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def _handle_generic_analysis(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle generic analysis for unknown database types"""
        
        tables = await self.mcp_client.list_tables()
        
        if not tables:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": "No tables found",
                "data": []
            }
        
        # Get sample data from first table
        table_name = tables[0]
        data = await self.mcp_client.get_sample_data(table_name, limit=10)
        
        return {
            "agent_name": self.agent_name,
            "success": True,
            "data": data,
            "metadata": {
                "record_count": len(data),
                "query_type": "generic_analysis",
                "sample_table": table_name,
                "available_tables": tables
            }
        }