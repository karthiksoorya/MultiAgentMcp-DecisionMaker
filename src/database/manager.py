# Advanced Database Manager with Connection Pooling and Performance Optimization
import asyncio
import asyncpg
from typing import Dict, Any, List, Optional, Union
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import time
import json
from datetime import datetime, timedelta

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool
except ImportError:
    # Fallback if SQLAlchemy is not available
    create_async_engine = None
    AsyncSession = None
    sessionmaker = None
    QueuePool = None

from ..utils.config import Config, DatabaseConfig
from ..utils.monitoring import track_database_operation

logger = logging.getLogger(__name__)

@dataclass
class ConnectionStats:
    """Database connection statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    average_response_time: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    last_connection_time: Optional[datetime] = None
    connection_history: List[dict] = field(default_factory=list)

@dataclass 
class QueryResult:
    """Enhanced query result with metadata"""
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query_hash: str
    database_name: str
    table_names: List[str]
    columns: List[str]
    success: bool
    error: Optional[str] = None
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class ConnectionPool:
    """Advanced connection pool with monitoring and optimization"""
    
    def __init__(self, db_config: DatabaseConfig, pool_size: int = 10, max_overflow: int = 20):
        self.db_config = db_config
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool = None
        self.stats = ConnectionStats()
        self._engine = None
        
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            # Create asyncpg pool
            self.pool = await asyncpg.create_pool(
                host=self.db_config.host,
                port=self.db_config.port,
                user=self.db_config.username,
                password=self.db_config.password,
                database=self.db_config.database,
                ssl='require' if self.db_config.ssl else 'prefer',
                min_size=2,
                max_size=self.pool_size,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0,
                command_timeout=60,
                server_settings={
                    'application_name': 'multi-agent-postgres-analysis',
                    'search_path': 'public'
                }
            )
            
            # Also create SQLAlchemy engine if available for advanced features
            if create_async_engine:
                database_url = f"postgresql+asyncpg://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.database}"
                self._engine = create_async_engine(
                    database_url,
                    poolclass=QueuePool,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
            
            self.stats.last_connection_time = datetime.now()
            logger.info(f"Connection pool initialized for {self.db_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool for {self.db_config.name}: {e}")
            self.stats.failed_connections += 1
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        connection = None
        start_time = time.time()
        
        try:
            if not self.pool:
                raise RuntimeError("Connection pool not initialized")
                
            connection = await self.pool.acquire()
            self.stats.active_connections += 1
            self.stats.total_connections += 1
            
            yield connection
            
        except Exception as e:
            self.stats.failed_connections += 1
            logger.error(f"Connection error: {e}")
            raise
        finally:
            if connection:
                await self.pool.release(connection)
                self.stats.active_connections = max(0, self.stats.active_connections - 1)
                
                # Update response time stats
                response_time = time.time() - start_time
                self.stats.average_response_time = (
                    (self.stats.average_response_time * (self.stats.total_connections - 1) + response_time) 
                    / self.stats.total_connections
                )
    
    async def execute_query(self, query: str, params: Optional[List] = None) -> QueryResult:
        """Execute query with enhanced monitoring and error handling"""
        start_time = time.time()
        query_hash = str(hash(query))
        
        try:
            async with self.get_connection() as connection:
                # Execute query
                if params:
                    rows = await connection.fetch(query, *params)
                else:
                    rows = await connection.fetch(query)
                
                # Convert to dictionaries
                data = [dict(row) for row in rows]
                execution_time = time.time() - start_time
                
                # Extract metadata
                columns = list(rows[0].keys()) if rows else []
                table_names = self._extract_table_names(query)
                
                result = QueryResult(
                    data=data,
                    row_count=len(data),
                    execution_time=execution_time,
                    query_hash=query_hash,
                    database_name=self.db_config.name,
                    table_names=table_names,
                    columns=columns,
                    success=True
                )
                
                self.stats.successful_queries += 1
                self.stats.total_queries += 1
                
                # Track for monitoring
                track_database_operation(
                    self.db_config.name, 
                    "query", 
                    execution_time, 
                    len(data),
                    True
                )
                
                return result
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.stats.failed_queries += 1
            self.stats.total_queries += 1
            
            # Track failed operation
            track_database_operation(
                self.db_config.name, 
                "query", 
                execution_time, 
                0,
                False
            )
            
            return QueryResult(
                data=[],
                row_count=0,
                execution_time=execution_time,
                query_hash=query_hash,
                database_name=self.db_config.name,
                table_names=[],
                columns=[],
                success=False,
                error=str(e)
            )
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from SQL query (basic implementation)"""
        import re
        
        # Simple regex to find table names after FROM and JOIN
        pattern = r'\b(?:FROM|JOIN)\s+(\w+)'
        matches = re.findall(pattern, query.upper())
        return list(set(matches))
    
    async def get_schema_info(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed schema information"""
        try:
            async with self.get_connection() as connection:
                if table_name:
                    # Get specific table schema
                    query = """
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length,
                        numeric_precision,
                        numeric_scale
                    FROM information_schema.columns 
                    WHERE table_name = $1 
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                    """
                    columns = await connection.fetch(query, table_name)
                    
                    # Get table statistics
                    stats_query = f"""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE tablename = $1
                    """
                    stats = await connection.fetch(stats_query, table_name)
                    
                    return {
                        "table_name": table_name,
                        "columns": [dict(col) for col in columns],
                        "statistics": [dict(stat) for stat in stats],
                        "database": self.db_config.name
                    }
                else:
                    # Get all tables
                    tables_query = """
                    SELECT 
                        table_name,
                        table_type,
                        is_insertable_into,
                        is_typed
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                    """
                    tables = await connection.fetch(tables_query)
                    
                    return {
                        "database": self.db_config.name,
                        "tables": [dict(table) for table in tables],
                        "table_count": len(tables)
                    }
                    
        except Exception as e:
            logger.error(f"Schema info error for {self.db_config.name}: {e}")
            return {"error": str(e), "database": self.db_config.name}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test database connection with detailed diagnostics"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as connection:
                # Test basic connectivity
                version_result = await connection.fetchval("SELECT version()")
                
                # Test query performance
                perf_start = time.time()
                await connection.fetchval("SELECT 1")
                ping_time = (time.time() - perf_start) * 1000  # ms
                
                # Get database size
                size_query = """
                SELECT pg_size_pretty(pg_database_size(current_database())) as database_size
                """
                size_result = await connection.fetchval(size_query)
                
                total_time = time.time() - start_time
                
                return {
                    "success": True,
                    "database": self.db_config.name,
                    "host": self.db_config.host,
                    "port": self.db_config.port,
                    "version": version_result,
                    "ping_time_ms": round(ping_time, 2),
                    "total_test_time": round(total_time, 3),
                    "database_size": size_result,
                    "pool_stats": {
                        "total_connections": self.stats.total_connections,
                        "active_connections": self.stats.active_connections,
                        "failed_connections": self.stats.failed_connections,
                        "avg_response_time": round(self.stats.average_response_time, 3)
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "database": self.db_config.name,
                "error": str(e),
                "test_duration": time.time() - start_time
            }
    
    async def cleanup(self):
        """Clean up connection pool"""
        if self.pool:
            await self.pool.close()
            
        if self._engine:
            await self._engine.dispose()
            
        logger.info(f"Connection pool cleaned up for {self.db_config.name}")

class EnhancedDatabaseManager:
    """Advanced database manager with multi-database support and monitoring"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pools: Dict[str, ConnectionPool] = {}
        self.global_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_databases": 0,
            "active_databases": 0,
            "startup_time": datetime.now()
        }
        
    async def initialize(self) -> bool:
        """Initialize all database connections"""
        success_count = 0
        
        for db_name, db_config in self.config.DATABASES.items():
            try:
                pool = ConnectionPool(db_config)
                if await pool.initialize():
                    self.pools[db_name] = pool
                    success_count += 1
                    logger.info(f"Initialized database pool: {db_name}")
                else:
                    logger.error(f"Failed to initialize database pool: {db_name}")
                    
            except Exception as e:
                logger.error(f"Database initialization error for {db_name}: {e}")
        
        self.global_stats["total_databases"] = len(self.config.DATABASES)
        self.global_stats["active_databases"] = success_count
        
        if success_count > 0:
            logger.info(f"Database manager initialized: {success_count}/{len(self.config.DATABASES)} databases")
            return True
        else:
            logger.error("No databases initialized successfully")
            return False
    
    async def execute_query(self, database_name: str, query: str, params: Optional[List] = None) -> QueryResult:
        """Execute query on specific database"""
        if database_name not in self.pools:
            return QueryResult(
                data=[],
                row_count=0,
                execution_time=0.0,
                query_hash="",
                database_name=database_name,
                table_names=[],
                columns=[],
                success=False,
                error=f"Database {database_name} not available"
            )
        
        pool = self.pools[database_name]
        result = await pool.execute_query(query, params)
        
        # Update global stats
        self.global_stats["total_queries"] += 1
        if result.success:
            self.global_stats["successful_queries"] += 1
        else:
            self.global_stats["failed_queries"] += 1
        
        return result
    
    async def execute_multi_database_query(self, query_plan: Dict[str, str], params: Optional[Dict[str, List]] = None) -> Dict[str, QueryResult]:
        """Execute queries across multiple databases concurrently"""
        tasks = []
        
        for db_name, query in query_plan.items():
            if db_name in self.pools:
                db_params = params.get(db_name) if params else None
                task = self.execute_query(db_name, query, db_params)
                tasks.append((db_name, task))
        
        results = {}
        if tasks:
            # Execute all queries concurrently
            task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (db_name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    results[db_name] = QueryResult(
                        data=[],
                        row_count=0,
                        execution_time=0.0,
                        query_hash="",
                        database_name=db_name,
                        table_names=[],
                        columns=[],
                        success=False,
                        error=str(result)
                    )
                else:
                    results[db_name] = result
        
        return results
    
    async def get_all_schemas(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get schema information for all or specific databases"""
        if database_name:
            if database_name in self.pools:
                return await self.pools[database_name].get_schema_info()
            else:
                return {"error": f"Database {database_name} not found"}
        
        # Get schemas for all databases
        schemas = {}
        tasks = []
        
        for db_name, pool in self.pools.items():
            tasks.append((db_name, pool.get_schema_info()))
        
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (db_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    schemas[db_name] = {"error": str(result)}
                else:
                    schemas[db_name] = result
        
        return schemas
    
    async def test_all_connections(self, include_details: bool = False) -> Dict[str, Any]:
        """Test all database connections"""
        results = {}
        tasks = []
        
        for db_name, pool in self.pools.items():
            tasks.append((db_name, pool.test_connection()))
        
        if tasks:
            test_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (db_name, _), result in zip(tasks, test_results):
                if isinstance(result, Exception):
                    results[db_name] = {
                        "success": False,
                        "error": str(result),
                        "database": db_name
                    }
                else:
                    results[db_name] = result if include_details else {"success": result["success"], "database": db_name}
        
        return {
            "results": results,
            "summary": {
                "total_databases": len(self.pools),
                "successful_connections": sum(1 for r in results.values() if r.get("success", False)),
                "failed_connections": sum(1 for r in results.values() if not r.get("success", False)),
                "test_timestamp": datetime.now().isoformat()
            }
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global database manager statistics"""
        uptime = datetime.now() - self.global_stats["startup_time"]
        
        pool_stats = {}
        for db_name, pool in self.pools.items():
            pool_stats[db_name] = {
                "total_connections": pool.stats.total_connections,
                "active_connections": pool.stats.active_connections,
                "failed_connections": pool.stats.failed_connections,
                "total_queries": pool.stats.total_queries,
                "successful_queries": pool.stats.successful_queries,
                "failed_queries": pool.stats.failed_queries,
                "avg_response_time": pool.stats.average_response_time
            }
        
        return {
            **self.global_stats,
            "uptime_seconds": uptime.total_seconds(),
            "success_rate": (
                self.global_stats["successful_queries"] / max(self.global_stats["total_queries"], 1) * 100
            ),
            "pool_stats": pool_stats
        }
    
    async def cleanup(self):
        """Cleanup all database connections"""
        cleanup_tasks = []
        
        for pool in self.pools.values():
            cleanup_tasks.append(pool.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self.pools.clear()
        logger.info("Database manager cleanup completed")

# Utility functions for easy access
async def create_database_manager(config: Optional[Config] = None) -> EnhancedDatabaseManager:
    """Create and initialize database manager"""
    config = config or Config()
    manager = EnhancedDatabaseManager(config)
    await manager.initialize()
    return manager