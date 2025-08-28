"""
Distributed Multi-Agent Orchestrator
Coordinates multiple specialized database agents to handle complex cross-database queries
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
from dataclasses import dataclass

from .distributed_agents import UsersAgent, ProductsAgent, SalesAgent

logger = logging.getLogger(__name__)

@dataclass
class QueryPlan:
    """Represents a query execution plan"""
    original_query: str
    sub_queries: List[Dict[str, Any]]
    required_agents: List[str]
    execution_order: List[str]
    requires_joining: bool
    join_keys: List[str]
    estimated_complexity: int

@dataclass
class AgentResult:
    """Represents result from an individual agent"""
    agent_name: str
    query_type: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None

class DistributedOrchestrator:
    """
    Orchestrates queries across multiple specialized database agents
    Handles query parsing, splitting, coordination, and result joining
    """
    
    def __init__(self):
        self.agents = {}
        self.query_patterns = self._initialize_query_patterns()
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_execution_time": 0.0
        }
    
    async def initialize(self):
        """Initialize all agents"""
        try:
            # Initialize specialized agents
            self.agents["users"] = UsersAgent()
            self.agents["products"] = ProductsAgent()
            self.agents["sales"] = SalesAgent()
            
            # Initialize each agent
            initialization_results = []
            for agent_name, agent in self.agents.items():
                result = await agent.initialize()
                initialization_results.append((agent_name, result))
                logger.info(f"Agent {agent_name} initialized: {result}")
            
            # Check if all agents initialized successfully
            all_success = all(result[1] for result in initialization_results)
            if all_success:
                logger.info("All agents initialized successfully")
                return True
            else:
                failed_agents = [name for name, success in initialization_results if not success]
                logger.error(f"Failed to initialize agents: {failed_agents}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    def _initialize_query_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regex patterns for query analysis"""
        return {
            "user_spending": {
                "pattern": r"(?:user|customer|member).+(?:spent|spending|spend|purchase|money|amount)",
                "agents": ["users", "sales"],
                "join_required": True,
                "join_key": "user_id"
            },
            "regional_analysis": {
                "pattern": r"(?:region|location|area|city|state).+(?:user|customer|product|sales|revenue)",
                "agents": ["users", "products", "sales"],
                "join_required": True,
                "join_key": "region"
            },
            "product_preferences": {
                "pattern": r"(?:product|item).+(?:preferred|popular|favorite|liked|purchased|bought)",
                "agents": ["products", "sales"],
                "join_required": True,
                "join_key": "product_id"
            },
            "sales_analysis": {
                "pattern": r"(?:sales|revenue|profit|income|earnings).+(?:region|product|user|customer)",
                "agents": ["sales", "users", "products"],
                "join_required": True,
                "join_key": "multiple"
            },
            "user_demographics": {
                "pattern": r"(?:user|customer).+(?:demographics|age|gender|location|profile)",
                "agents": ["users"],
                "join_required": False,
                "join_key": None
            },
            "product_catalog": {
                "pattern": r"(?:product|item|catalog).+(?:category|brand|price|inventory)",
                "agents": ["products"],
                "join_required": False,
                "join_key": None
            }
        }
    
    async def process_query(self, original_query: str) -> Dict[str, Any]:
        """
        Main method to process a complex query
        Example: "How much users spent more and on which region and which products are preferred more?"
        """
        start_time = datetime.now()
        self.execution_stats["total_queries"] += 1
        
        try:
            # Step 1: Analyze and create query plan
            logger.info(f"Processing query: {original_query}")
            query_plan = await self._create_query_plan(original_query)
            
            # Step 2: Execute sub-queries with appropriate agents
            agent_results = await self._execute_distributed_queries(query_plan)
            
            # Step 3: Join and correlate results from different agents
            consolidated_result = await self._consolidate_results(agent_results, query_plan)
            
            # Step 4: Generate insights and recommendations
            final_result = await self._generate_insights(consolidated_result, query_plan)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_stats["successful_queries"] += 1
            self._update_avg_execution_time(execution_time)
            
            return {
                "success": True,
                "original_query": original_query,
                "query_plan": {
                    "required_agents": query_plan.required_agents,
                    "execution_order": query_plan.execution_order,
                    "requires_joining": query_plan.requires_joining,
                    "complexity": query_plan.estimated_complexity
                },
                "agent_results": [
                    {
                        "agent": result.agent_name,
                        "query_type": result.query_type,
                        "success": result.success,
                        "data_count": len(result.data) if result.data else 0,
                        "execution_time": result.execution_time
                    }
                    for result in agent_results
                ],
                "consolidated_data": final_result["data"],
                "insights": final_result["insights"],
                "recommendations": final_result.get("recommendations", []),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.execution_stats["failed_queries"] += 1
            logger.error(f"Query processing failed: {e}")
            
            return {
                "success": False,
                "original_query": original_query,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_query_plan(self, query: str) -> QueryPlan:
        """Create an execution plan for the complex query"""
        query_lower = query.lower()
        required_agents = []
        join_keys = []
        requires_joining = False
        
        # Analyze query to determine which agents are needed
        for pattern_name, pattern_config in self.query_patterns.items():
            if re.search(pattern_config["pattern"], query_lower, re.IGNORECASE):
                required_agents.extend(pattern_config["agents"])
                if pattern_config["join_required"]:
                    requires_joining = True
                    if pattern_config["join_key"]:
                        join_keys.append(pattern_config["join_key"])
        
        # Remove duplicates and determine execution order
        required_agents = list(set(required_agents))
        
        # Determine optimal execution order (users first, then products, then sales)
        execution_order = []
        if "users" in required_agents:
            execution_order.append("users")
        if "products" in required_agents:
            execution_order.append("products")  
        if "sales" in required_agents:
            execution_order.append("sales")
        
        # Create sub-queries for each agent
        sub_queries = []
        for agent_name in execution_order:
            sub_queries.append({
                "agent": agent_name,
                "query": query,  # Each agent will interpret the query in their context
                "context": {
                    "requires_cross_reference": [a for a in required_agents if a != agent_name],
                    "join_keys": join_keys
                }
            })
        
        # Estimate complexity
        complexity = len(required_agents) * (2 if requires_joining else 1)
        
        return QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            required_agents=required_agents,
            execution_order=execution_order,
            requires_joining=requires_joining,
            join_keys=join_keys,
            estimated_complexity=complexity
        )
    
    async def _execute_distributed_queries(self, query_plan: QueryPlan) -> List[AgentResult]:
        """Execute queries across multiple agents"""
        results = []
        
        # Execute queries in parallel for better performance
        tasks = []
        for sub_query in query_plan.sub_queries:
            agent_name = sub_query["agent"]
            agent = self.agents[agent_name]
            task = self._execute_agent_query(agent, sub_query["query"], sub_query["context"])
            tasks.append(task)
        
        # Wait for all agent queries to complete
        agent_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, response in enumerate(agent_responses):
            if isinstance(response, Exception):
                results.append(AgentResult(
                    agent_name=query_plan.sub_queries[i]["agent"],
                    query_type="error",
                    data=[],
                    metadata={},
                    execution_time=0.0,
                    success=False,
                    error=str(response)
                ))
            else:
                results.append(response)
        
        return results
    
    async def _execute_agent_query(self, agent, query: str, context: Dict[str, Any]) -> AgentResult:
        """Execute query on a specific agent"""
        start_time = datetime.now()
        
        try:
            result = await agent.process_query(query, context)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=agent.agent_name,
                query_type=result.get("query_type", "unknown"),
                data=result.get("data", []),
                metadata=result.get("metadata", {}),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Agent {agent.agent_name} query failed: {e}")
            
            return AgentResult(
                agent_name=agent.agent_name,
                query_type="error",
                data=[],
                metadata={},
                execution_time=execution_time,
                success=False,
                error=str(e)
            )
    
    async def _consolidate_results(self, agent_results: List[AgentResult], query_plan: QueryPlan) -> Dict[str, Any]:
        """Consolidate results from multiple agents"""
        consolidated = {
            "users_data": [],
            "products_data": [],
            "sales_data": [],
            "joined_data": [],
            "summary": {}
        }
        
        # Organize data by agent type
        for result in agent_results:
            if result.success:
                if result.agent_name == "UsersAgent":
                    consolidated["users_data"] = result.data
                elif result.agent_name == "ProductsAgent":
                    consolidated["products_data"] = result.data
                elif result.agent_name == "SalesAgent":
                    consolidated["sales_data"] = result.data
        
        # If joining is required, perform cross-database joins
        if query_plan.requires_joining:
            consolidated["joined_data"] = await self._perform_data_joins(consolidated, query_plan.join_keys)
        
        # Generate summary statistics
        consolidated["summary"] = {
            "total_users": len(consolidated["users_data"]),
            "total_products": len(consolidated["products_data"]),
            "total_transactions": len(consolidated["sales_data"]),
            "agents_used": [r.agent_name for r in agent_results if r.success],
            "processing_time": sum(r.execution_time for r in agent_results)
        }
        
        return consolidated
    
    async def _perform_data_joins(self, consolidated_data: Dict[str, Any], join_keys: List[str]) -> List[Dict[str, Any]]:
        """Perform joins between data from different databases"""
        joined_data = []
        
        users_data = consolidated_data["users_data"]
        products_data = consolidated_data["products_data"]
        sales_data = consolidated_data["sales_data"]
        
        # Example join logic for the query: "How much users spent more and on which region and which products are preferred more?"
        try:
            # Create lookup dictionaries for efficient joining
            users_lookup = {user.get("user_id"): user for user in users_data} if users_data else {}
            products_lookup = {product.get("product_id"): product for product in products_data} if products_data else {}
            
            # Join sales data with user and product data
            for sale in sales_data:
                joined_record = {
                    "transaction_data": sale,
                    "user_data": users_lookup.get(sale.get("user_id"), {}),
                    "product_data": {}  # Would need product_id in sales data to join
                }
                
                # Add regional information
                if "region" in sale:
                    joined_record["region"] = sale["region"]
                elif joined_record["user_data"] and "region" in joined_record["user_data"]:
                    joined_record["region"] = joined_record["user_data"]["region"]
                
                joined_data.append(joined_record)
            
        except Exception as e:
            logger.error(f"Data join failed: {e}")
            # Return partial data if join fails
            return []
        
        return joined_data
    
    async def _generate_insights(self, consolidated_result: Dict[str, Any], query_plan: QueryPlan) -> Dict[str, Any]:
        """Generate insights from consolidated data"""
        insights = []
        recommendations = []
        
        try:
            # Analyze spending patterns by region
            if consolidated_result["sales_data"]:
                regional_spending = {}
                for sale in consolidated_result["sales_data"]:
                    region = sale.get("region", "Unknown")
                    amount = float(sale.get("total_spent", 0) or sale.get("amount", 0) or 0)
                    
                    if region not in regional_spending:
                        regional_spending[region] = {"total": 0, "transactions": 0, "users": set()}
                    
                    regional_spending[region]["total"] += amount
                    regional_spending[region]["transactions"] += 1
                    if sale.get("user_id"):
                        regional_spending[region]["users"].add(sale["user_id"])
                
                # Generate regional insights
                if regional_spending:
                    top_region = max(regional_spending.items(), key=lambda x: x[1]["total"])
                    insights.append({
                        "type": "regional_analysis",
                        "insight": f"Highest spending region is {top_region[0]} with total spending of ${top_region[1]['total']:,.2f}",
                        "data": dict(regional_spending)
                    })
            
            # Analyze user spending patterns
            if consolidated_result["users_data"] and consolidated_result["sales_data"]:
                high_spenders = []
                user_spending = {}
                
                for sale in consolidated_result["sales_data"]:
                    user_id = sale.get("user_id")
                    amount = float(sale.get("total_spent", 0) or sale.get("amount", 0) or 0)
                    
                    if user_id:
                        if user_id not in user_spending:
                            user_spending[user_id] = 0
                        user_spending[user_id] += amount
                
                # Find top spenders
                if user_spending:
                    sorted_spenders = sorted(user_spending.items(), key=lambda x: x[1], reverse=True)
                    top_spenders = sorted_spenders[:5]  # Top 5 spenders
                    
                    insights.append({
                        "type": "user_spending",
                        "insight": f"Top spender: User ID {top_spenders[0][0]} with ${top_spenders[0][1]:,.2f} in total spending",
                        "top_spenders": top_spenders
                    })
            
            # Product preference analysis
            if consolidated_result["products_data"]:
                product_insights = []
                for product in consolidated_result["products_data"][:5]:  # Top 5 products
                    product_insights.append({
                        "product_name": product.get("product_name", "Unknown"),
                        "category": product.get("category_name", "Unknown"),
                        "price": product.get("price", 0),
                        "rating": product.get("avg_rating", 0)
                    })
                
                if product_insights:
                    insights.append({
                        "type": "product_preferences",
                        "insight": f"Most preferred products span across {len(set(p['category'] for p in product_insights))} categories",
                        "top_products": product_insights
                    })
            
            # Generate recommendations
            recommendations.extend([
                "Focus marketing efforts on the highest spending region",
                "Develop loyalty programs for top spending users",
                "Expand inventory of highly-rated products",
                "Analyze seasonal spending patterns for better forecasting"
            ])
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            insights.append({
                "type": "error",
                "insight": "Unable to generate detailed insights due to data processing error",
                "error": str(e)
            })
        
        return {
            "data": consolidated_result,
            "insights": insights,
            "recommendations": recommendations
        }
    
    def _update_avg_execution_time(self, execution_time: float):
        """Update average execution time statistics"""
        total_queries = self.execution_stats["total_queries"]
        current_avg = self.execution_stats["avg_execution_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_queries - 1)) + execution_time) / total_queries
        self.execution_stats["avg_execution_time"] = new_avg
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator and agent status"""
        agent_status = {}
        for name, agent in self.agents.items():
            agent_status[name] = {
                "agent_name": agent.agent_name,
                "mcp_port": agent.mcp_port,
                "database_path": agent.database_path,
                "capabilities": agent.capabilities,
                "tables": await agent._get_table_list()
            }
        
        return {
            "orchestrator_stats": self.execution_stats,
            "agents": agent_status,
            "query_patterns": list(self.query_patterns.keys())
        }
    
    async def cleanup(self):
        """Cleanup orchestrator and all agents"""
        logger.info("Cleaning up distributed orchestrator...")
        # In a full implementation, this would cleanup agent connections
        self.agents.clear()