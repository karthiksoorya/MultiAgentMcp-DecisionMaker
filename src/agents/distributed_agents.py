"""
Distributed Database Agents
Each agent is specialized for a specific database and connects to its dedicated MCP server
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDatabaseAgent(ABC):
    """Base class for all database agents"""
    
    def __init__(self, agent_name: str, mcp_port: int, database_path: str):
        self.agent_name = agent_name
        self.mcp_port = mcp_port
        self.database_path = database_path
        self.mcp_client = None
        self.capabilities = []
        
    async def initialize(self):
        """Initialize the agent and verify MCP server connection"""
        try:
            # In a real implementation, this would connect to the MCP server
            # For now, we'll simulate the connection
            logger.info(f"{self.agent_name} initialized with MCP server on port {self.mcp_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_name}: {e}")
            return False
    
    @abstractmethod
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query specific to this agent's domain"""
        pass
    
    @abstractmethod
    def can_handle_query(self, query: str) -> bool:
        """Determine if this agent can handle the given query"""
        pass
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information"""
        return {
            "agent": self.agent_name,
            "database": self.database_path,
            "tables": await self._get_table_list(),
            "capabilities": self.capabilities
        }
    
    async def _get_table_list(self) -> List[str]:
        """Get list of tables in the database"""
        # This would call the MCP server's list_tables tool
        # For now, return mock data based on agent type
        return []
    
    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool call"""
        try:
            # In real implementation, this would make HTTP calls to MCP server
            # For now, simulate the response
            logger.info(f"{self.agent_name} executing {tool_name} with args: {arguments}")
            return {
                "success": True,
                "agent": self.agent_name,
                "tool": tool_name,
                "result": "Mock result - would connect to actual MCP server"
            }
        except Exception as e:
            logger.error(f"MCP tool execution failed in {self.agent_name}: {e}")
            return {"success": False, "error": str(e)}

class UsersAgent(BaseDatabaseAgent):
    """Agent specialized for Users database operations"""
    
    def __init__(self):
        super().__init__(
            agent_name="UsersAgent", 
            mcp_port=8001,
            database_path="./data/users_db.sqlite"
        )
        self.capabilities = [
            "user_demographics", "user_behavior", "user_segmentation",
            "registration_patterns", "activity_analysis", "regional_distribution"
        ]
        self.user_keywords = [
            "user", "users", "customer", "customers", "member", "members",
            "registration", "signup", "demographics", "profile", "account"
        ]
    
    def can_handle_query(self, query: str) -> bool:
        """Check if query relates to users/customers"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.user_keywords)
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user-related queries"""
        query_lower = query.lower()
        
        # Determine query type and generate appropriate SQL
        if "spent" in query_lower or "spending" in query_lower:
            return await self._analyze_user_spending(query, context)
        elif "region" in query_lower:
            return await self._analyze_user_regions(query, context)
        elif "demographics" in query_lower:
            return await self._analyze_user_demographics(query, context)
        elif "active" in query_lower:
            return await self._analyze_active_users(query, context)
        else:
            return await self._general_user_query(query, context)
    
    async def _analyze_user_spending(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user spending patterns"""
        sql_query = """
            SELECT u.user_id, u.username, u.first_name, u.last_name, 
                   ua.region, u.user_type, u.status
            FROM users u
            LEFT JOIN user_addresses ua ON u.user_id = ua.user_id
            WHERE u.status = 'active'
            ORDER BY u.user_type DESC, u.created_at DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "user_spending_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "requires_cross_reference": ["sales_data"],
                "user_segments": ["premium", "regular"],
                "analysis_type": "spending_patterns"
            }
        }
    
    async def _analyze_user_regions(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze users by region"""
        sql_query = """
            SELECT ua.region, COUNT(u.user_id) as user_count,
                   COUNT(CASE WHEN u.user_type = 'premium' THEN 1 END) as premium_users,
                   COUNT(CASE WHEN u.status = 'active' THEN 1 END) as active_users
            FROM users u
            LEFT JOIN user_addresses ua ON u.user_id = ua.user_id
            GROUP BY ua.region
            ORDER BY user_count DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "regional_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "regions_covered": ["West Coast", "Midwest", "Southeast", "Northeast", "Mountain", "Southwest"],
                "analysis_type": "regional_distribution"
            }
        }
    
    async def _analyze_user_demographics(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user demographics"""
        sql_query = """
            SELECT 
                CASE 
                    WHEN (julianday('now') - julianday(date_of_birth)) / 365.25 < 25 THEN '18-24'
                    WHEN (julianday('now') - julianday(date_of_birth)) / 365.25 < 35 THEN '25-34'
                    WHEN (julianday('now') - julianday(date_of_birth)) / 365.25 < 45 THEN '35-44'
                    ELSE '45+'
                END as age_group,
                u.user_type,
                ua.region,
                COUNT(*) as user_count
            FROM users u
            LEFT JOIN user_addresses ua ON u.user_id = ua.user_id
            WHERE u.date_of_birth IS NOT NULL
            GROUP BY age_group, u.user_type, ua.region
            ORDER BY user_count DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "demographic_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "age_groups": ["18-24", "25-34", "35-44", "45+"],
                "user_types": ["premium", "regular"]
            }
        }
    
    async def _analyze_active_users(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze active users"""
        sql_query = """
            SELECT u.user_id, u.username, u.first_name, u.last_name,
                   u.user_type, ua.region, u.last_login, u.created_at
            FROM users u
            LEFT JOIN user_addresses ua ON u.user_id = ua.user_id
            WHERE u.status = 'active'
            ORDER BY u.last_login DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "active_users_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "filter": "active_status",
                "sort_by": "last_login"
            }
        }
    
    async def _general_user_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general user queries"""
        sql_query = """
            SELECT u.*, ua.region, ua.city, ua.state
            FROM users u
            LEFT JOIN user_addresses ua ON u.user_id = ua.user_id
            WHERE u.status = 'active'
            LIMIT 100
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "general_user_query",
            "sql_executed": sql_query,
            "data": result.get("result", [])
        }
    
    async def _get_table_list(self) -> List[str]:
        return ["users", "user_addresses", "user_preferences"]

class ProductsAgent(BaseDatabaseAgent):
    """Agent specialized for Products database operations"""
    
    def __init__(self):
        super().__init__(
            agent_name="ProductsAgent",
            mcp_port=8002,
            database_path="./data/products_db.sqlite"
        )
        self.capabilities = [
            "product_catalog", "inventory_management", "pricing_analysis",
            "category_analysis", "product_performance", "review_analysis"
        ]
        self.product_keywords = [
            "product", "products", "item", "items", "catalog", "inventory",
            "category", "categories", "brand", "brands", "price", "pricing",
            "preferred", "popular", "review", "rating"
        ]
    
    def can_handle_query(self, query: str) -> bool:
        """Check if query relates to products"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.product_keywords)
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process product-related queries"""
        query_lower = query.lower()
        
        if "preferred" in query_lower or "popular" in query_lower:
            return await self._analyze_product_preferences(query, context)
        elif "category" in query_lower:
            return await self._analyze_product_categories(query, context)
        elif "region" in query_lower:
            return await self._analyze_products_by_region(query, context)
        elif "price" in query_lower or "pricing" in query_lower:
            return await self._analyze_product_pricing(query, context)
        else:
            return await self._general_product_query(query, context)
    
    async def _analyze_product_preferences(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product preferences and popularity"""
        sql_query = """
            SELECT p.product_id, p.product_name, p.brand, c.category_name,
                   p.price, AVG(pr.rating) as avg_rating,
                   COUNT(pr.review_id) as review_count,
                   SUM(i.quantity_available) as total_inventory
            FROM products p
            LEFT JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
            LEFT JOIN inventory i ON p.product_id = i.product_id
            WHERE p.status = 'active'
            GROUP BY p.product_id, p.product_name, p.brand, c.category_name, p.price
            ORDER BY avg_rating DESC, review_count DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "product_preferences",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "requires_cross_reference": ["sales_data"],
                "analysis_type": "product_popularity",
                "sort_criteria": ["rating", "review_count"]
            }
        }
    
    async def _analyze_product_categories(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze products by category"""
        sql_query = """
            SELECT c.category_name, 
                   COUNT(p.product_id) as product_count,
                   AVG(p.price) as avg_price,
                   MIN(p.price) as min_price,
                   MAX(p.price) as max_price,
                   SUM(i.quantity_available) as total_inventory
            FROM categories c
            LEFT JOIN products p ON c.category_id = p.category_id
            LEFT JOIN inventory i ON p.product_id = i.product_id
            WHERE p.status = 'active'
            GROUP BY c.category_name
            ORDER BY product_count DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "category_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "categories": ["Electronics", "Clothing", "Home & Garden", "Sports & Outdoors", "Books", "Health & Beauty"]
            }
        }
    
    async def _analyze_products_by_region(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product availability by region"""
        sql_query = """
            SELECT i.region, p.product_name, c.category_name, p.brand,
                   i.quantity_available, i.warehouse_location, p.price
            FROM inventory i
            JOIN products p ON i.product_id = p.product_id
            JOIN categories c ON p.category_id = c.category_id
            WHERE i.quantity_available > 0 AND p.status = 'active'
            ORDER BY i.region, i.quantity_available DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "regional_inventory",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "requires_cross_reference": ["sales_by_region"],
                "analysis_type": "regional_availability"
            }
        }
    
    async def _analyze_product_pricing(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze product pricing"""
        sql_query = """
            SELECT c.category_name, p.brand,
                   COUNT(p.product_id) as product_count,
                   AVG(p.price) as avg_price,
                   MIN(p.price) as min_price,
                   MAX(p.price) as max_price,
                   (p.price - p.cost) as profit_margin
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            WHERE p.status = 'active' AND p.cost IS NOT NULL
            GROUP BY c.category_name, p.brand
            ORDER BY avg_price DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "pricing_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", [])
        }
    
    async def _general_product_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general product queries"""
        sql_query = """
            SELECT p.*, c.category_name, AVG(pr.rating) as avg_rating
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            LEFT JOIN product_reviews pr ON p.product_id = pr.product_id
            WHERE p.status = 'active'
            GROUP BY p.product_id
            ORDER BY p.created_at DESC
            LIMIT 50
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "general_product_query",
            "sql_executed": sql_query,
            "data": result.get("result", [])
        }
    
    async def _get_table_list(self) -> List[str]:
        return ["products", "categories", "inventory", "product_reviews"]

class SalesAgent(BaseDatabaseAgent):
    """Agent specialized for Sales database operations"""
    
    def __init__(self):
        super().__init__(
            agent_name="SalesAgent",
            mcp_port=8003,
            database_path="./data/sales_db.sqlite"
        )
        self.capabilities = [
            "sales_analytics", "revenue_analysis", "order_processing",
            "transaction_analysis", "regional_sales", "payment_analysis"
        ]
        self.sales_keywords = [
            "sales", "sale", "order", "orders", "transaction", "transactions",
            "revenue", "payment", "purchase", "bought", "spent", "spending",
            "total", "amount", "money", "profit"
        ]
    
    def can_handle_query(self, query: str) -> bool:
        """Check if query relates to sales/transactions"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.sales_keywords)
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process sales-related queries"""
        query_lower = query.lower()
        
        if "spent" in query_lower or "spending" in query_lower:
            return await self._analyze_spending_patterns(query, context)
        elif "region" in query_lower:
            return await self._analyze_sales_by_region(query, context)
        elif "revenue" in query_lower or "total" in query_lower:
            return await self._analyze_revenue(query, context)
        elif "order" in query_lower:
            return await self._analyze_orders(query, context)
        else:
            return await self._general_sales_query(query, context)
    
    async def _analyze_spending_patterns(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user spending patterns"""
        sql_query = """
            SELECT t.user_id, 
                   SUM(t.amount) as total_spent,
                   COUNT(t.transaction_id) as transaction_count,
                   AVG(t.amount) as avg_transaction,
                   t.region,
                   MAX(t.transaction_date) as last_purchase
            FROM transactions t
            WHERE t.payment_status = 'completed'
            GROUP BY t.user_id, t.region
            ORDER BY total_spent DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "spending_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "requires_cross_reference": ["user_data"],
                "analysis_type": "user_spending",
                "sort_by": "total_spent"
            }
        }
    
    async def _analyze_sales_by_region(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sales by region"""
        sql_query = """
            SELECT o.region,
                   COUNT(o.order_id) as total_orders,
                   SUM(o.total_amount) as total_revenue,
                   AVG(o.total_amount) as avg_order_value,
                   COUNT(DISTINCT o.user_id) as unique_customers
            FROM orders o
            WHERE o.order_status IN ('completed', 'shipped')
            GROUP BY o.region
            ORDER BY total_revenue DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "regional_sales",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "regions": ["West Coast", "Midwest", "Southeast", "Northeast", "Mountain", "Southwest"],
                "metrics": ["revenue", "orders", "avg_order_value", "unique_customers"]
            }
        }
    
    async def _analyze_revenue(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze revenue and financial metrics"""
        sql_query = """
            SELECT 
                DATE(o.order_date) as order_date,
                o.region,
                SUM(o.total_amount) as daily_revenue,
                COUNT(o.order_id) as daily_orders,
                AVG(o.total_amount) as avg_order_value,
                SUM(o.tax_amount) as total_tax,
                SUM(o.shipping_amount) as total_shipping
            FROM orders o
            WHERE o.order_status IN ('completed', 'shipped')
            GROUP BY DATE(o.order_date), o.region
            ORDER BY order_date DESC, daily_revenue DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "revenue_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "time_period": "daily",
                "financial_metrics": ["revenue", "tax", "shipping", "avg_order_value"]
            }
        }
    
    async def _analyze_orders(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze order patterns"""
        sql_query = """
            SELECT o.order_id, o.user_id, o.order_date, o.order_status,
                   o.total_amount, o.region, o.payment_method,
                   oi.product_id, oi.quantity, oi.unit_price
            FROM orders o
            JOIN order_items oi ON o.order_id = oi.order_id
            WHERE o.order_date >= DATE('now', '-30 days')
            ORDER BY o.order_date DESC
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "order_analysis",
            "sql_executed": sql_query,
            "data": result.get("result", []),
            "metadata": {
                "requires_cross_reference": ["product_data", "user_data"],
                "time_filter": "last_30_days"
            }
        }
    
    async def _general_sales_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general sales queries"""
        sql_query = """
            SELECT t.*, o.order_status, o.payment_method
            FROM transactions t
            LEFT JOIN orders o ON t.order_id = o.order_id
            WHERE t.payment_status = 'completed'
            ORDER BY t.transaction_date DESC
            LIMIT 100
        """
        
        result = await self._execute_mcp_tool("read_query", {"query": sql_query})
        
        return {
            "agent": self.agent_name,
            "query_type": "general_sales_query", 
            "sql_executed": sql_query,
            "data": result.get("result", [])
        }
    
    async def _get_table_list(self) -> List[str]:
        return ["orders", "order_items", "transactions", "sales_analytics"]