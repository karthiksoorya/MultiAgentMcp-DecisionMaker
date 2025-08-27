"""
Database tools adapted from executeautomation/mcp-database-server
Provides standardized MCP database operations integrated with the existing Python architecture
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import asyncpg
import csv
import io

from ..utils.config import Config
from ..database.manager import ConnectionPool

logger = logging.getLogger(__name__)

class DatabaseTools:
    """
    Database tools compatible with MCP protocol standards
    Adapted from executeautomation/mcp-database-server
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.connection_pools = {}
        self.insights = []
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize connection pools for all configured databases
            for db_name, db_config in self.config.databases.items():
                pool = ConnectionPool(db_config)
                await pool.initialize()
                self.connection_pools[db_name] = pool
                logger.info(f"Initialized connection pool for database: {db_name}")
        except Exception as e:
            logger.error(f"Failed to initialize database tools: {e}")
            raise
    
    async def read_query(self, query: str, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute SELECT queries to read data from the database
        Compatible with MCP read_query tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            # Validate it's a SELECT query
            if not query.strip().upper().startswith('SELECT'):
                raise ValueError("read_query only supports SELECT statements")
            
            async with pool.pool.acquire() as conn:
                start_time = datetime.now()
                rows = await conn.fetch(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Convert rows to list of dictionaries
                results = [dict(row) for row in rows]
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "query": query,
                                "database": db_name,
                                "row_count": len(results),
                                "execution_time_seconds": execution_time,
                                "results": results
                            }, indent=2, default=str)
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error executing read query: {e}")
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": json.dumps({
                            "error": str(e),
                            "query": query,
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def write_query(self, query: str, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute INSERT, UPDATE, or DELETE queries
        Compatible with MCP write_query tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            # Validate it's a write query
            query_upper = query.strip().upper()
            if not any(query_upper.startswith(op) for op in ['INSERT', 'UPDATE', 'DELETE']):
                raise ValueError("write_query only supports INSERT, UPDATE, and DELETE statements")
            
            async with pool.pool.acquire() as conn:
                start_time = datetime.now()
                result = await conn.execute(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "query": query,
                                "database": db_name,
                                "result": result,
                                "execution_time_seconds": execution_time,
                                "status": "success"
                            }, indent=2)
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error executing write query: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "query": query,
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def create_table(self, query: str, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Create new tables in the database
        Compatible with MCP create_table tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            # Validate it's a CREATE TABLE query
            if not query.strip().upper().startswith('CREATE TABLE'):
                raise ValueError("create_table requires a CREATE TABLE statement")
            
            async with pool.pool.acquire() as conn:
                start_time = datetime.now()
                result = await conn.execute(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "query": query,
                                "database": db_name,
                                "result": result,
                                "execution_time_seconds": execution_time,
                                "status": "table_created"
                            }, indent=2)
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error creating table: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "query": query,
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def list_tables(self, database: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a list of all tables in the database
        Compatible with MCP list_tables tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            async with pool.pool.acquire() as conn:
                # PostgreSQL query to list tables
                query = """
                    SELECT table_name, table_schema 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """
                rows = await conn.fetch(query)
                tables = [{"name": row["table_name"], "schema": row["table_schema"]} for row in rows]
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "database": db_name,
                                "table_count": len(tables),
                                "tables": tables
                            }, indent=2)
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def describe_table(self, table_name: str, database: Optional[str] = None) -> Dict[str, Any]:
        """
        View schema information for a specific table
        Compatible with MCP describe_table tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            async with pool.pool.acquire() as conn:
                # PostgreSQL query to describe table structure
                query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = $1 AND table_schema = 'public'
                    ORDER BY ordinal_position
                """
                rows = await conn.fetch(query, table_name)
                columns = [dict(row) for row in rows]
                
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "database": db_name,
                                "table_name": table_name,
                                "column_count": len(columns),
                                "columns": columns
                            }, indent=2)
                        }
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error describing table: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "table_name": table_name,
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def export_query(self, query: str, format: str = "json", database: Optional[str] = None) -> Dict[str, Any]:
        """
        Export query results to various formats (CSV, JSON)
        Compatible with MCP export_query tool
        """
        try:
            # Use first database if none specified
            db_name = database or next(iter(self.connection_pools.keys()))
            pool = self.connection_pools.get(db_name)
            
            if not pool:
                raise ValueError(f"Database '{db_name}' not found")
            
            # Validate format
            if format.lower() not in ['json', 'csv']:
                raise ValueError("Format must be 'json' or 'csv'")
            
            async with pool.pool.acquire() as conn:
                rows = await conn.fetch(query)
                results = [dict(row) for row in rows]
                
                if format.lower() == 'csv':
                    # Convert to CSV
                    if results:
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                        csv_data = output.getvalue()
                        output.close()
                        
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": csv_data
                                }
                            ]
                        }
                    else:
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "No data returned by query"
                                }
                            ]
                        }
                else:
                    # JSON format
                    return {
                        "content": [
                            {
                                "type": "text", 
                                "text": json.dumps({
                                    "query": query,
                                    "database": db_name,
                                    "row_count": len(results),
                                    "format": format,
                                    "data": results
                                }, indent=2, default=str)
                            }
                        ]
                    }
                
        except Exception as e:
            logger.error(f"Error exporting query: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "query": query,
                            "format": format,
                            "database": database
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def append_insight(self, insight: str) -> Dict[str, Any]:
        """
        Add a business insight to the memo
        Compatible with MCP append_insight tool
        """
        try:
            insight_entry = {
                "id": len(self.insights) + 1,
                "insight": insight,
                "timestamp": datetime.now().isoformat(),
                "source": "mcp_database_tools"
            }
            
            self.insights.append(insight_entry)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "status": "insight_added",
                            "insight_id": insight_entry["id"],
                            "insight": insight,
                            "total_insights": len(self.insights)
                        }, indent=2)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error adding insight: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "insight": insight
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def list_insights(self) -> Dict[str, Any]:
        """
        List all business insights in the memo
        Compatible with MCP list_insights tool
        """
        try:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "total_insights": len(self.insights),
                            "insights": self.insights
                        }, indent=2, default=str)
                    }
                ]
            }
            
        except Exception as e:
            logger.error(f"Error listing insights: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e)
                        }, indent=2)
                    }
                ],
                "isError": True
            }
    
    async def cleanup(self):
        """Cleanup database connections"""
        for db_name, pool in self.connection_pools.items():
            try:
                await pool.cleanup()
                logger.info(f"Cleaned up connection pool for database: {db_name}")
            except Exception as e:
                logger.error(f"Error cleaning up database {db_name}: {e}")