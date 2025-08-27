# Advanced Performance Monitoring with Prometheus Integration
import time
import logging
import functools
from typing import Dict, Any, Callable, Optional, List
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback implementations
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args): pass
        def labels(self, *args): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args): pass
        def labels(self, *args): return self
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args): pass
        def inc(self, *args): pass
        def dec(self, *args): pass
        def labels(self, *args): return self

logger = logging.getLogger(__name__)

class EnhancedMetricsCollector:
    """Advanced metrics collection with Prometheus integration"""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Core metrics
        self.query_metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "query_history": deque(maxlen=1000)  # Keep last 1000 queries
        }
        
        # Agent performance tracking
        self.agent_metrics = defaultdict(lambda: {
            "executions": 0,
            "total_time": 0.0,
            "total_records": 0,
            "errors": 0,
            "last_execution": None,
            "execution_history": deque(maxlen=100)
        })
        
        # Database performance tracking
        self.database_metrics = defaultdict(lambda: {
            "connections": 0,
            "queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "connection_errors": 0,
            "last_activity": None
        })
        
        # System-wide metrics
        self.system_metrics = {
            "startup_time": self.start_time,
            "active_sessions": 0,
            "peak_concurrent_queries": 0,
            "current_concurrent_queries": 0,
            "memory_usage": 0,
            "cpu_usage": 0
        }
        
        # MCP-specific metrics
        self.mcp_metrics = {
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "average_tool_response_time": 0.0,
            "tool_call_history": deque(maxlen=500)
        }
        
        # Initialize Prometheus metrics if available
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Counters
        self.prom_queries_total = Counter(
            'multiagent_queries_total',
            'Total number of queries executed',
            ['database', 'status'],
            registry=self.registry
        )
        
        self.prom_agent_executions = Counter(
            'multiagent_agent_executions_total',
            'Total agent executions',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.prom_mcp_calls = Counter(
            'multiagent_mcp_calls_total',
            'Total MCP tool calls',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        # Histograms for timing
        self.prom_query_duration = Histogram(
            'multiagent_query_duration_seconds',
            'Query execution time in seconds',
            ['database'],
            registry=self.registry
        )
        
        self.prom_agent_duration = Histogram(
            'multiagent_agent_duration_seconds',
            'Agent execution time in seconds',
            ['agent_name'],
            registry=self.registry
        )
        
        self.prom_mcp_duration = Histogram(
            'multiagent_mcp_duration_seconds',
            'MCP tool call duration in seconds',
            ['tool_name'],
            registry=self.registry
        )
        
        # Gauges
        self.prom_active_connections = Gauge(
            'multiagent_active_connections',
            'Number of active database connections',
            ['database'],
            registry=self.registry
        )
        
        self.prom_concurrent_queries = Gauge(
            'multiagent_concurrent_queries',
            'Current number of concurrent queries',
            registry=self.registry
        )
        
        self.prom_system_uptime = Gauge(
            'multiagent_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
    
    def record_query(self, database: str, query: str, execution_time: float, success: bool, record_count: int = 0):
        """Record query execution metrics"""
        with self.lock:
            self.query_metrics["total_queries"] += 1
            self.query_metrics["total_execution_time"] += execution_time
            
            if success:
                self.query_metrics["successful_queries"] += 1
            else:
                self.query_metrics["failed_queries"] += 1
            
            self.query_metrics["average_execution_time"] = (
                self.query_metrics["total_execution_time"] / self.query_metrics["total_queries"]
            )
            
            # Store query history
            query_record = {
                "timestamp": datetime.now(),
                "database": database,
                "query_hash": str(hash(query)),
                "execution_time": execution_time,
                "success": success,
                "record_count": record_count
            }
            self.query_metrics["query_history"].append(query_record)
            
            # Update database-specific metrics
            db_metrics = self.database_metrics[database]
            db_metrics["queries"] += 1
            db_metrics["total_execution_time"] += execution_time
            db_metrics["last_activity"] = datetime.now()
            
            if success:
                db_metrics["successful_queries"] += 1
            else:
                db_metrics["failed_queries"] += 1
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            status = "success" if success else "failure"
            self.prom_queries_total.labels(database=database, status=status).inc()
            self.prom_query_duration.labels(database=database).observe(execution_time)
    
    def record_agent_performance(self, agent_name: str, execution_time: float, record_count: int, success: bool = True):
        """Record agent-specific performance"""
        with self.lock:
            agent_metrics = self.agent_metrics[agent_name]
            agent_metrics["executions"] += 1
            agent_metrics["total_time"] += execution_time
            agent_metrics["total_records"] += record_count
            agent_metrics["last_execution"] = datetime.now()
            
            if not success:
                agent_metrics["errors"] += 1
            
            # Store execution history
            execution_record = {
                "timestamp": datetime.now(),
                "execution_time": execution_time,
                "record_count": record_count,
                "success": success
            }
            agent_metrics["execution_history"].append(execution_record)
            
            # Calculate averages
            agent_metrics["average_time"] = agent_metrics["total_time"] / agent_metrics["executions"]
            agent_metrics["average_records"] = agent_metrics["total_records"] / agent_metrics["executions"]
            agent_metrics["success_rate"] = ((agent_metrics["executions"] - agent_metrics["errors"]) / agent_metrics["executions"]) * 100
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            status = "success" if success else "failure"
            self.prom_agent_executions.labels(agent_name=agent_name, status=status).inc()
            self.prom_agent_duration.labels(agent_name=agent_name).observe(execution_time)
    
    def record_mcp_call(self, tool_name: str, success: bool, execution_time: float):
        """Record MCP tool call metrics"""
        with self.lock:
            self.mcp_metrics["tool_calls"] += 1
            
            if success:
                self.mcp_metrics["successful_tool_calls"] += 1
            else:
                self.mcp_metrics["failed_tool_calls"] += 1
            
            # Update average response time
            total_time = self.mcp_metrics["average_tool_response_time"] * (self.mcp_metrics["tool_calls"] - 1)
            self.mcp_metrics["average_tool_response_time"] = (total_time + execution_time) / self.mcp_metrics["tool_calls"]
            
            # Store call history
            call_record = {
                "timestamp": datetime.now(),
                "tool_name": tool_name,
                "execution_time": execution_time,
                "success": success
            }
            self.mcp_metrics["tool_call_history"].append(call_record)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            status = "success" if success else "failure"
            self.prom_mcp_calls.labels(tool_name=tool_name, status=status).inc()
            self.prom_mcp_duration.labels(tool_name=tool_name).observe(execution_time)
    
    def get_comprehensive_metrics(self, include_history: bool = True) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Update uptime metric
            if PROMETHEUS_AVAILABLE:
                self.prom_system_uptime.set(uptime)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": uptime,
                "system": {
                    **self.system_metrics,
                    "uptime_formatted": str(timedelta(seconds=int(uptime)))
                },
                "queries": {
                    **self.query_metrics,
                    "success_rate": (
                        (self.query_metrics["successful_queries"] / max(self.query_metrics["total_queries"], 1)) * 100
                    ),
                    "queries_per_second": self.query_metrics["total_queries"] / max(uptime, 1)
                },
                "agents": dict(self.agent_metrics),
                "databases": dict(self.database_metrics),
                "mcp": {
                    **self.mcp_metrics,
                    "success_rate": (
                        (self.mcp_metrics["successful_tool_calls"] / max(self.mcp_metrics["tool_calls"], 1)) * 100
                    )
                }
            }
            
            # Include history if requested
            if not include_history:
                metrics["queries"].pop("query_history", None)
                metrics["mcp"].pop("tool_call_history", None)
                for agent_data in metrics["agents"].values():
                    agent_data.pop("execution_history", None)
            
            return metrics

# Global metrics collector instance
metrics_collector = EnhancedMetricsCollector()

# Convenience functions
def track_database_operation(database: str, operation: str, execution_time: float, record_count: int, success: bool):
    """Track database operation"""
    metrics_collector.record_query(database, operation, execution_time, success, record_count)

def track_agent_execution(agent_name: str, execution_time: float, record_count: int, success: bool = True):
    """Track agent execution"""
    metrics_collector.record_agent_performance(agent_name, execution_time, record_count, success)

def track_mcp_call(tool_name: str, success: bool, execution_time: float):
    """Track MCP tool call"""
    metrics_collector.record_mcp_call(tool_name, success, execution_time)

def get_metrics(include_history: bool = True) -> Dict[str, Any]:
    """Get comprehensive metrics"""
    return metrics_collector.get_comprehensive_metrics(include_history)