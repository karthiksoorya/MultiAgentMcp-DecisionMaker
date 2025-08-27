# Specialized Database Agents for User, Transaction, and Analytics
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod

from ..database.manager import EnhancedDatabaseManager, ConnectionPool, QueryResult
from ..utils.config import DatabaseConfig
from ..utils.llm_client import MultiProviderLLMClient
from ..utils.monitoring import track_agent_execution

logger = logging.getLogger(__name__)

class BaseSpecializedAgent(ABC):
    """Base class for specialized database agents"""
    
    def __init__(self, name: str, llm_client: Optional[MultiProviderLLMClient] = None):
        self.name = name
        self.llm_client = llm_client
        self.execution_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "last_execution": None
        }
    
    @abstractmethod
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform specialized analysis"""
        pass
    
    def _update_stats(self, execution_time: float, success: bool):
        """Update execution statistics"""
        self.execution_stats["total_analyses"] += 1
        self.execution_stats["total_execution_time"] += execution_time
        self.execution_stats["last_execution"] = datetime.now()
        
        if success:
            self.execution_stats["successful_analyses"] += 1
        
        if self.execution_stats["total_analyses"] > 0:
            self.execution_stats["average_execution_time"] = (
                self.execution_stats["total_execution_time"] / self.execution_stats["total_analyses"]
            )

class UserAnalysisAgent(BaseSpecializedAgent):
    """Specialized agent for user data analysis"""
    
    def __init__(self, database_name: str, db_config: DatabaseConfig, llm_client: Optional[MultiProviderLLMClient] = None):
        super().__init__(f"user_analysis_{database_name}", llm_client)
        self.database_name = database_name
        self.db_config = db_config
        self.connection_pool = None
        
        # User analysis capabilities
        self.capabilities = {
            "user_demographics": True,
            "user_behavior_analysis": True,
            "user_segmentation": True,
            "user_lifecycle_analysis": True,
            "churn_prediction": True if llm_client else False
        }
    
    async def initialize(self) -> bool:
        """Initialize the user analysis agent"""
        try:
            self.connection_pool = ConnectionPool(self.db_config)
            success = await self.connection_pool.initialize()
            if success:
                logger.info(f"User Analysis Agent initialized for {self.database_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize User Analysis Agent: {e}")
            return False
    
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive user analysis"""
        start_time = datetime.now()
        
        try:
            # Analyze query to determine specific user analysis needed
            analysis_type = self._determine_analysis_type(query)
            
            # Execute appropriate analysis
            if analysis_type == "user_demographics":
                result = await self._analyze_user_demographics(query, context)
            elif analysis_type == "user_behavior":
                result = await self._analyze_user_behavior(query, context)
            elif analysis_type == "user_segmentation":
                result = await self._analyze_user_segmentation(query, context)
            elif analysis_type == "user_lifecycle":
                result = await self._analyze_user_lifecycle(query, context)
            else:
                # General user analysis
                result = await self._analyze_general_users(query, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, result.get("success", False))
            
            # Track execution
            track_agent_execution(
                self.name,
                execution_time,
                len(result.get("data", [])),
                result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, False)
            
            logger.error(f"User analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {"execution_time": execution_time, "agent": self.name}
            }
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of user analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["demographic", "age", "gender", "location"]):
            return "user_demographics"
        elif any(word in query_lower for word in ["behavior", "activity", "engagement", "usage"]):
            return "user_behavior"
        elif any(word in query_lower for word in ["segment", "group", "category", "cluster"]):
            return "user_segmentation"
        elif any(word in query_lower for word in ["lifecycle", "journey", "retention", "churn"]):
            return "user_lifecycle"
        else:
            return "general"
    
    async def _analyze_user_demographics(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze user demographics"""
        sql_query = """
        SELECT 
            user_id,
            username,
            email,
            created_at,
            last_login,
            status,
            EXTRACT(YEAR FROM AGE(created_at)) as account_age_years,
            CASE 
                WHEN last_login > NOW() - INTERVAL '7 days' THEN 'very_active'
                WHEN last_login > NOW() - INTERVAL '30 days' THEN 'active'
                WHEN last_login > NOW() - INTERVAL '90 days' THEN 'inactive'
                ELSE 'dormant'
            END as activity_level
        FROM users 
        WHERE status IN ('active', 'inactive')
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        query_result = await self.connection_pool.execute_query(sql_query)
        
        if query_result.success:
            # Generate demographic insights
            demographics = self._calculate_demographics(query_result.data)
            
            return {
                "success": True,
                "analysis_type": "user_demographics",
                "data": query_result.data,
                "demographics": demographics,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "user_ids": [row["user_id"] for row in query_result.data]
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    async def _analyze_user_behavior(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        sql_query = """
        SELECT 
            user_id,
            username,
            created_at,
            last_login,
            status,
            CASE 
                WHEN last_login IS NULL THEN 0
                ELSE EXTRACT(EPOCH FROM (last_login - created_at)) / 86400
            END as days_to_first_login,
            CASE 
                WHEN last_login IS NULL THEN 0
                ELSE EXTRACT(EPOCH FROM (NOW() - last_login)) / 86400
            END as days_since_last_login
        FROM users 
        WHERE created_at >= NOW() - INTERVAL '1 year'
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        query_result = await self.connection_pool.execute_query(sql_query)
        
        if query_result.success:
            # Analyze behavior patterns
            behavior_analysis = self._analyze_behavior_patterns(query_result.data)
            
            return {
                "success": True,
                "analysis_type": "user_behavior",
                "data": query_result.data,
                "behavior_analysis": behavior_analysis,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "user_ids": [row["user_id"] for row in query_result.data]
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    async def _analyze_user_segmentation(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Segment users based on various criteria"""
        sql_query = """
        SELECT 
            user_id,
            username,
            email,
            created_at,
            last_login,
            status,
            EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400 as account_age_days,
            CASE 
                WHEN last_login > NOW() - INTERVAL '7 days' THEN 'highly_active'
                WHEN last_login > NOW() - INTERVAL '30 days' THEN 'active'
                WHEN last_login > NOW() - INTERVAL '90 days' THEN 'at_risk'
                ELSE 'churned'
            END as segment
        FROM users 
        WHERE status != 'deleted'
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        query_result = await self.connection_pool.execute_query(sql_query)
        
        if query_result.success:
            # Perform segmentation analysis
            segmentation = self._perform_user_segmentation(query_result.data)
            
            return {
                "success": True,
                "analysis_type": "user_segmentation",
                "data": query_result.data,
                "segmentation": segmentation,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "user_ids": [row["user_id"] for row in query_result.data]
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    async def _analyze_user_lifecycle(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze user lifecycle and retention"""
        sql_query = """
        SELECT 
            user_id,
            username,
            created_at,
            last_login,
            status,
            EXTRACT(EPOCH FROM (COALESCE(last_login, NOW()) - created_at)) / 86400 as lifecycle_days,
            CASE 
                WHEN last_login IS NULL THEN 'never_returned'
                WHEN last_login > NOW() - INTERVAL '7 days' THEN 'active'
                WHEN last_login > NOW() - INTERVAL '30 days' THEN 'recent'
                WHEN last_login > NOW() - INTERVAL '90 days' THEN 'declining'
                ELSE 'churned'
            END as lifecycle_stage
        FROM users 
        WHERE created_at >= NOW() - INTERVAL '6 months'
        ORDER BY created_at DESC
        LIMIT 1000
        """
        
        query_result = await self.connection_pool.execute_query(sql_query)
        
        if query_result.success:
            # Analyze lifecycle patterns
            lifecycle_analysis = self._analyze_lifecycle_patterns(query_result.data)
            
            return {
                "success": True,
                "analysis_type": "user_lifecycle",
                "data": query_result.data,
                "lifecycle_analysis": lifecycle_analysis,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "user_ids": [row["user_id"] for row in query_result.data]
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    async def _analyze_general_users(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """General user analysis"""
        sql_query = """
        SELECT 
            user_id,
            username,
            email,
            created_at,
            last_login,
            status
        FROM users 
        WHERE status = 'active'
        ORDER BY created_at DESC
        LIMIT 500
        """
        
        query_result = await self.connection_pool.execute_query(sql_query)
        
        if query_result.success:
            return {
                "success": True,
                "analysis_type": "general_user_analysis",
                "data": query_result.data,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "user_ids": [row["user_id"] for row in query_result.data]
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    def _calculate_demographics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate demographic statistics"""
        if not data:
            return {}
        
        total_users = len(data)
        activity_levels = {}
        account_ages = []
        
        for record in data:
            # Activity level distribution
            activity = record.get("activity_level", "unknown")
            activity_levels[activity] = activity_levels.get(activity, 0) + 1
            
            # Account age
            if "account_age_years" in record:
                account_ages.append(record["account_age_years"])
        
        return {
            "total_users": total_users,
            "activity_distribution": {
                level: (count / total_users * 100) for level, count in activity_levels.items()
            },
            "average_account_age_years": sum(account_ages) / len(account_ages) if account_ages else 0
        }
    
    def _analyze_behavior_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        if not data:
            return {}
        
        total_users = len(data)
        never_logged_in = 0
        active_users = 0
        
        days_to_login = []
        days_since_login = []
        
        for record in data:
            if not record.get("last_login"):
                never_logged_in += 1
            else:
                if record.get("days_since_last_login", 0) < 30:
                    active_users += 1
                
                if record.get("days_to_first_login"):
                    days_to_login.append(record["days_to_first_login"])
                
                if record.get("days_since_last_login"):
                    days_since_login.append(record["days_since_last_login"])
        
        return {
            "total_users": total_users,
            "never_logged_in_percentage": (never_logged_in / total_users * 100),
            "active_users_percentage": (active_users / total_users * 100),
            "average_days_to_first_login": sum(days_to_login) / len(days_to_login) if days_to_login else 0,
            "average_days_since_last_login": sum(days_since_login) / len(days_since_login) if days_since_login else 0
        }
    
    def _perform_user_segmentation(self, data: List[Dict]) -> Dict[str, Any]:
        """Perform user segmentation analysis"""
        if not data:
            return {}
        
        segments = {}
        total_users = len(data)
        
        for record in data:
            segment = record.get("segment", "unknown")
            segments[segment] = segments.get(segment, 0) + 1
        
        return {
            "total_users": total_users,
            "segments": {
                segment: {
                    "count": count,
                    "percentage": (count / total_users * 100)
                } for segment, count in segments.items()
            }
        }
    
    def _analyze_lifecycle_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze user lifecycle patterns"""
        if not data:
            return {}
        
        lifecycle_stages = {}
        total_users = len(data)
        lifecycle_days = []
        
        for record in data:
            stage = record.get("lifecycle_stage", "unknown")
            lifecycle_stages[stage] = lifecycle_stages.get(stage, 0) + 1
            
            if record.get("lifecycle_days"):
                lifecycle_days.append(record["lifecycle_days"])
        
        return {
            "total_users": total_users,
            "lifecycle_stages": {
                stage: {
                    "count": count,
                    "percentage": (count / total_users * 100)
                } for stage, count in lifecycle_stages.items()
            },
            "average_lifecycle_days": sum(lifecycle_days) / len(lifecycle_days) if lifecycle_days else 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.connection_pool:
            await self.connection_pool.cleanup()

class TransactionAnalysisAgent(BaseSpecializedAgent):
    """Specialized agent for transaction data analysis"""
    
    def __init__(self, database_name: str, db_config: DatabaseConfig, llm_client: Optional[MultiProviderLLMClient] = None):
        super().__init__(f"transaction_analysis_{database_name}", llm_client)
        self.database_name = database_name
        self.db_config = db_config
        self.connection_pool = None
        
        # Transaction analysis capabilities
        self.capabilities = {
            "transaction_patterns": True,
            "spending_analysis": True,
            "fraud_detection": True if llm_client else False,
            "merchant_analysis": True,
            "temporal_analysis": True
        }
    
    async def initialize(self) -> bool:
        """Initialize the transaction analysis agent"""
        try:
            self.connection_pool = ConnectionPool(self.db_config)
            success = await self.connection_pool.initialize()
            if success:
                logger.info(f"Transaction Analysis Agent initialized for {self.database_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize Transaction Analysis Agent: {e}")
            return False
    
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive transaction analysis"""
        start_time = datetime.now()
        
        try:
            # Determine analysis type
            analysis_type = self._determine_analysis_type(query)
            
            # Use context data if available (e.g., user IDs from user analysis)
            user_ids = context.get("user_ids", []) if context else []
            
            # Execute appropriate analysis
            if analysis_type == "spending_patterns":
                result = await self._analyze_spending_patterns(query, user_ids)
            elif analysis_type == "merchant_analysis":
                result = await self._analyze_merchant_patterns(query, user_ids)
            elif analysis_type == "temporal_analysis":
                result = await self._analyze_temporal_patterns(query, user_ids)
            elif analysis_type == "fraud_detection":
                result = await self._analyze_fraud_patterns(query, user_ids)
            else:
                result = await self._analyze_general_transactions(query, user_ids)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, result.get("success", False))
            
            # Track execution
            track_agent_execution(
                self.name,
                execution_time,
                len(result.get("data", [])),
                result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, False)
            
            logger.error(f"Transaction analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {"execution_time": execution_time, "agent": self.name}
            }
    
    def _determine_analysis_type(self, query: str) -> str:
        """Determine the type of transaction analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["spending", "amount", "spend", "value"]):
            return "spending_patterns"
        elif any(word in query_lower for word in ["merchant", "store", "vendor", "business"]):
            return "merchant_analysis"
        elif any(word in query_lower for word in ["time", "temporal", "trend", "daily", "monthly"]):
            return "temporal_analysis"
        elif any(word in query_lower for word in ["fraud", "suspicious", "anomaly", "unusual"]):
            return "fraud_detection"
        else:
            return "general"
    
    async def _analyze_spending_patterns(self, query: str, user_ids: List = None) -> Dict[str, Any]:
        """Analyze spending patterns"""
        if user_ids:
            placeholders = ",".join([f"${i+1}" for i in range(len(user_ids))])
            sql_query = f"""
            SELECT 
                user_id,
                transaction_date,
                amount,
                category,
                merchant,
                EXTRACT(DOW FROM transaction_date) as day_of_week,
                EXTRACT(HOUR FROM transaction_date) as hour_of_day
            FROM transactions 
            WHERE user_id IN ({placeholders})
            AND transaction_date >= NOW() - INTERVAL '3 months'
            ORDER BY transaction_date DESC
            LIMIT 2000
            """
            params = user_ids
        else:
            sql_query = """
            SELECT 
                user_id,
                transaction_date,
                amount,
                category,
                merchant,
                EXTRACT(DOW FROM transaction_date) as day_of_week,
                EXTRACT(HOUR FROM transaction_date) as hour_of_day
            FROM transactions 
            WHERE transaction_date >= NOW() - INTERVAL '1 month'
            ORDER BY transaction_date DESC
            LIMIT 1000
            """
            params = None
        
        query_result = await self.connection_pool.execute_query(sql_query, params)
        
        if query_result.success:
            spending_analysis = self._calculate_spending_patterns(query_result.data)
            
            return {
                "success": True,
                "analysis_type": "spending_patterns",
                "data": query_result.data,
                "spending_analysis": spending_analysis,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "used_user_context": bool(user_ids),
                    "context_user_count": len(user_ids) if user_ids else 0
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    async def _analyze_general_transactions(self, query: str, user_ids: List = None) -> Dict[str, Any]:
        """General transaction analysis"""
        if user_ids:
            placeholders = ",".join([f"${i+1}" for i in range(min(len(user_ids), 100))])  # Limit for performance
            user_ids_limited = user_ids[:100]
            sql_query = f"""
            SELECT 
                user_id,
                transaction_date,
                amount,
                category,
                merchant
            FROM transactions 
            WHERE user_id IN ({placeholders})
            ORDER BY transaction_date DESC
            LIMIT 1000
            """
            params = user_ids_limited
        else:
            sql_query = """
            SELECT 
                user_id,
                transaction_date,
                amount,
                category,
                merchant
            FROM transactions 
            ORDER BY transaction_date DESC
            LIMIT 500
            """
            params = None
        
        query_result = await self.connection_pool.execute_query(sql_query, params)
        
        if query_result.success:
            return {
                "success": True,
                "analysis_type": "general_transaction_analysis",
                "data": query_result.data,
                "metadata": {
                    "record_count": len(query_result.data),
                    "execution_time": query_result.execution_time,
                    "agent": self.name,
                    "used_user_context": bool(user_ids),
                    "context_user_count": len(user_ids) if user_ids else 0
                }
            }
        else:
            return {
                "success": False,
                "error": query_result.error,
                "data": []
            }
    
    def _calculate_spending_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate spending pattern statistics"""
        if not data:
            return {}
        
        total_transactions = len(data)
        total_amount = sum(float(record.get("amount", 0)) for record in data)
        
        # Category analysis
        category_spending = {}
        for record in data:
            category = record.get("category", "unknown")
            amount = float(record.get("amount", 0))
            category_spending[category] = category_spending.get(category, 0) + amount
        
        # Day of week analysis
        dow_spending = {}
        for record in data:
            dow = record.get("day_of_week", 0)
            amount = float(record.get("amount", 0))
            dow_spending[dow] = dow_spending.get(dow, 0) + amount
        
        return {
            "total_transactions": total_transactions,
            "total_amount": total_amount,
            "average_transaction_amount": total_amount / total_transactions if total_transactions > 0 else 0,
            "category_spending": category_spending,
            "day_of_week_spending": dow_spending,
            "top_categories": sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.connection_pool:
            await self.connection_pool.cleanup()

class AnalyticsAgent(BaseSpecializedAgent):
    """Advanced analytics agent for cross-database insights"""
    
    def __init__(self, llm_client: Optional[MultiProviderLLMClient], db_manager: Optional[EnhancedDatabaseManager] = None):
        super().__init__("advanced_analytics", llm_client)
        self.db_manager = db_manager
        
        # Analytics capabilities
        self.capabilities = {
            "cross_database_correlation": True,
            "trend_analysis": True,
            "predictive_insights": True if llm_client else False,
            "anomaly_detection": True,
            "statistical_analysis": True
        }
    
    async def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform advanced analytics on consolidated data"""
        start_time = datetime.now()
        
        try:
            # Extract data from context (from other agents)
            user_data = context.get("user_analysis", []) if context else []
            transaction_data = context.get("transaction_analysis", []) if context else []
            
            # Perform cross-database analytics
            if user_data and transaction_data:
                result = await self._perform_cross_database_analytics(query, user_data, transaction_data)
            elif user_data:
                result = await self._perform_user_analytics(query, user_data)
            elif transaction_data:
                result = await self._perform_transaction_analytics(query, transaction_data)
            else:
                result = {
                    "success": False,
                    "error": "No data available for analytics",
                    "data": []
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, result.get("success", False))
            
            # Track execution
            track_agent_execution(
                self.name,
                execution_time,
                len(result.get("analytics", {})),
                result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(execution_time, False)
            
            logger.error(f"Analytics analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "metadata": {"execution_time": execution_time, "agent": self.name}
            }
    
    async def _perform_cross_database_analytics(self, query: str, user_data: List, transaction_data: List) -> Dict[str, Any]:
        """Perform analytics across user and transaction data"""
        
        # Correlate user and transaction data
        user_transaction_map = {}
        
        # Build user map
        user_map = {record["user_id"]: record for record in user_data}
        
        # Map transactions to users
        for transaction in transaction_data:
            user_id = transaction.get("user_id")
            if user_id in user_map:
                if user_id not in user_transaction_map:
                    user_transaction_map[user_id] = {
                        "user": user_map[user_id],
                        "transactions": []
                    }
                user_transaction_map[user_id]["transactions"].append(transaction)
        
        # Perform analytics
        analytics = {
            "total_users_with_transactions": len(user_transaction_map),
            "average_transactions_per_user": 0,
            "total_transaction_value": 0,
            "user_segments_spending": {},
            "high_value_users": []
        }
        
        if user_transaction_map:
            total_transactions = sum(len(data["transactions"]) for data in user_transaction_map.values())
            total_value = sum(
                sum(float(t.get("amount", 0)) for t in data["transactions"])
                for data in user_transaction_map.values()
            )
            
            analytics["average_transactions_per_user"] = total_transactions / len(user_transaction_map)
            analytics["total_transaction_value"] = total_value
            
            # Find high-value users
            user_spending = []
            for user_id, data in user_transaction_map.items():
                user_total = sum(float(t.get("amount", 0)) for t in data["transactions"])
                user_spending.append({
                    "user_id": user_id,
                    "username": data["user"].get("username", "unknown"),
                    "total_spending": user_total,
                    "transaction_count": len(data["transactions"])
                })
            
            # Sort by spending and take top 10
            analytics["high_value_users"] = sorted(user_spending, key=lambda x: x["total_spending"], reverse=True)[:10]
        
        return {
            "success": True,
            "analysis_type": "cross_database_analytics",
            "data": list(user_transaction_map.values())[:100],  # Sample data
            "analytics": analytics,
            "metadata": {
                "total_users": len(user_data),
                "total_transactions": len(transaction_data),
                "correlated_users": len(user_transaction_map),
                "agent": self.name
            }
        }
    
    async def _perform_user_analytics(self, query: str, user_data: List) -> Dict[str, Any]:
        """Perform analytics on user data only"""
        analytics = {
            "total_users": len(user_data),
            "user_distribution": {},
            "activity_analysis": {}
        }
        
        # Analyze user status distribution
        status_count = {}
        for user in user_data:
            status = user.get("status", "unknown")
            status_count[status] = status_count.get(status, 0) + 1
        
        analytics["user_distribution"] = status_count
        
        return {
            "success": True,
            "analysis_type": "user_analytics",
            "data": user_data[:50],  # Sample
            "analytics": analytics,
            "metadata": {"agent": self.name}
        }
    
    async def _perform_transaction_analytics(self, query: str, transaction_data: List) -> Dict[str, Any]:
        """Perform analytics on transaction data only"""
        analytics = {
            "total_transactions": len(transaction_data),
            "total_value": sum(float(t.get("amount", 0)) for t in transaction_data),
            "average_transaction_value": 0,
            "category_analysis": {}
        }
        
        if transaction_data:
            analytics["average_transaction_value"] = analytics["total_value"] / len(transaction_data)
            
            # Category analysis
            category_count = {}
            for transaction in transaction_data:
                category = transaction.get("category", "unknown")
                category_count[category] = category_count.get(category, 0) + 1
            
            analytics["category_analysis"] = category_count
        
        return {
            "success": True,
            "analysis_type": "transaction_analytics",
            "data": transaction_data[:50],  # Sample
            "analytics": analytics,
            "metadata": {"agent": self.name}
        }