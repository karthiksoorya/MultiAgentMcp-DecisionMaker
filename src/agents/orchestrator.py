# Enhanced Orchestrator with Advanced LangGraph Workflows
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, TypedDict, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
import time

from ..utils.config import Config
from ..database.manager import EnhancedDatabaseManager, QueryResult
from ..utils.llm_client import MultiProviderLLMClient, LLMResponse
from ..utils.monitoring import track_agent_execution, get_metrics
from .specialized_agents import UserAnalysisAgent, TransactionAnalysisAgent, AnalyticsAgent

logger = logging.getLogger(__name__)

# Enhanced State for LangGraph
class EnhancedMultiAgentState(TypedDict):
    # Query information
    query: str
    session_id: str
    user_preferences: Dict[str, Any]
    
    # Planning phase
    analysis_plan: Dict[str, Any]
    execution_strategy: str
    required_agents: List[str]
    agent_dependencies: Dict[str, List[str]]
    
    # Execution phase
    agent_results: Dict[str, Any]
    consolidated_data: Dict[str, Any]
    data_correlations: List[Dict[str, Any]]
    
    # Insights generation
    insights: str
    confidence_score: float
    recommendations: List[str]
    
    # System state
    current_step: str
    execution_time: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class EnhancedOrchestrator:
    """Advanced orchestrator with intelligent workflow management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = None
        self.llm_client = None
        self.specialized_agents = {}
        self.workflow = None
        self.checkpointer = MemorySaver()  # For workflow state persistence
        
        # Performance tracking
        self.execution_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
    
    async def initialize(self) -> bool:
        """Initialize orchestrator and all components"""
        logger.info("Initializing Enhanced Orchestrator...")
        
        try:
            # Initialize database manager
            from ..database.manager import create_database_manager
            self.db_manager = await create_database_manager(self.config)
            
            # Initialize LLM client
            from ..utils.llm_client import create_llm_client
            self.llm_client = create_llm_client(self.config)
            
            # Initialize specialized agents
            await self._initialize_specialized_agents()
            
            # Create workflow
            self._create_workflow()
            
            logger.info("Enhanced Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Orchestrator: {e}")
            return False
    
    async def _initialize_specialized_agents(self):
        """Initialize specialized database agents"""
        # User Analysis Agent
        if "users_db" in self.config.DATABASES:
            user_agent = UserAnalysisAgent(
                "users_db", 
                self.config.DATABASES["users_db"], 
                self.llm_client
            )
            if await user_agent.initialize():
                self.specialized_agents["user_analysis"] = user_agent
                logger.info("User Analysis Agent initialized")
        
        # Transaction Analysis Agent  
        if "transactions_db" in self.config.DATABASES:
            transaction_agent = TransactionAnalysisAgent(
                "transactions_db",
                self.config.DATABASES["transactions_db"], 
                self.llm_client
            )
            if await transaction_agent.initialize():
                self.specialized_agents["transaction_analysis"] = transaction_agent
                logger.info("Transaction Analysis Agent initialized")
        
        # Analytics Agent (works with consolidated data)
        analytics_agent = AnalyticsAgent(self.llm_client, self.db_manager)
        self.specialized_agents["analytics"] = analytics_agent
        logger.info("Analytics Agent initialized")
    
    def _create_workflow(self):
        """Create advanced LangGraph workflow"""
        workflow = StateGraph(EnhancedMultiAgentState)
        
        # Add workflow nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("create_execution_plan", self.create_execution_plan)
        workflow.add_node("execute_agents", self.execute_agents)
        workflow.add_node("correlate_data", self.correlate_data)
        workflow.add_node("generate_insights", self.generate_insights)
        workflow.add_node("create_recommendations", self.create_recommendations)
        workflow.add_node("finalize_results", self.finalize_results)
        
        # Add conditional edges
        workflow.add_edge("analyze_query", "create_execution_plan")
        workflow.add_edge("create_execution_plan", "execute_agents")
        workflow.add_edge("execute_agents", "correlate_data")
        workflow.add_edge("correlate_data", "generate_insights")
        workflow.add_edge("generate_insights", "create_recommendations")
        workflow.add_edge("create_recommendations", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Compile with checkpointer for state persistence
        self.workflow = workflow.compile(checkpointer=self.checkpointer)
        logger.info("Advanced workflow created")
    
    async def process_query(
        self, 
        query: str, 
        include_insights: bool = True,
        max_records: int = 1000,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process query through enhanced workflow"""
        
        if not self.workflow:
            return {"success": False, "error": "Workflow not initialized"}
        
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()
        
        # Create initial state
        initial_state: EnhancedMultiAgentState = {
            "query": query,
            "session_id": session_id,
            "user_preferences": {"include_insights": include_insights, "max_records": max_records},
            "analysis_plan": {},
            "execution_strategy": "",
            "required_agents": [],
            "agent_dependencies": {},
            "agent_results": {},
            "consolidated_data": {},
            "data_correlations": [],
            "insights": "",
            "confidence_score": 0.0,
            "recommendations": [],
            "current_step": "starting",
            "execution_time": 0.0,
            "errors": [],
            "warnings": [],
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "orchestrator_version": "2.0",
                "available_agents": list(self.specialized_agents.keys())
            }
        }
        
        try:
            # Execute workflow
            config = {"configurable": {"thread_id": session_id}}
            result = await self.workflow.ainvoke(initial_state, config)
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            
            # Update stats
            self.execution_stats["total_queries"] += 1
            if result["current_step"] == "complete":
                self.execution_stats["successful_queries"] += 1
            
            # Track execution
            track_agent_execution(
                "enhanced_orchestrator",
                execution_time,
                len(result.get("consolidated_data", {}).get("records", [])),
                result["current_step"] == "complete"
            )
            
            return {
                "success": result["current_step"] == "complete",
                "session_id": session_id,
                "query": query,
                "execution_time": execution_time,
                "insights": result["insights"],
                "recommendations": result["recommendations"],
                "confidence_score": result["confidence_score"],
                "data": result["consolidated_data"],
                "correlations": result["data_correlations"],
                "execution_plan": result["analysis_plan"],
                "metadata": result["metadata"],
                "errors": result["errors"],
                "warnings": result["warnings"]
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed: {e}")
            
            track_agent_execution("enhanced_orchestrator", execution_time, 0, False)
            
            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
                "execution_time": execution_time
            }
    
    async def analyze_query(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Analyze user query to understand intent and requirements"""
        logger.info(f"Analyzing query: {state['query']}")
        
        try:
            # Use LLM to understand query intent
            analysis_prompt = f"""
            Analyze the following database query and determine:
            1. What type of analysis is needed (user analysis, transaction analysis, cross-database correlation, etc.)
            2. Which databases/tables are likely needed
            3. What specific data points should be retrieved
            4. Any potential challenges or dependencies
            
            Query: "{state['query']}"
            
            Available agents: {list(self.specialized_agents.keys())}
            Available databases: {list(self.config.DATABASES.keys())}
            
            Respond with a JSON analysis including:
            - intent: main purpose of the query
            - complexity: simple/moderate/complex
            - required_data_types: list of data types needed
            - suggested_approach: recommended analysis approach
            - potential_challenges: any foreseeable issues
            """
            
            llm_response = await self.llm_client.generate_completion(
                analysis_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            if llm_response.success:
                try:
                    analysis_result = json.loads(llm_response.content)
                except json.JSONDecodeError:
                    # Fallback analysis
                    analysis_result = {
                        "intent": "general_database_analysis",
                        "complexity": "moderate",
                        "required_data_types": ["user_data", "transaction_data"],
                        "suggested_approach": "sequential_analysis",
                        "potential_challenges": ["data_correlation"]
                    }
            else:
                state["warnings"].append("LLM query analysis failed, using fallback approach")
                analysis_result = {
                    "intent": "fallback_analysis",
                    "complexity": "unknown",
                    "required_data_types": ["unknown"],
                    "suggested_approach": "try_all_agents",
                    "potential_challenges": ["llm_unavailable"]
                }
            
            state["analysis_plan"]["query_analysis"] = analysis_result
            state["current_step"] = "query_analyzed"
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["errors"].append(f"Query analysis failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def create_execution_plan(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Create intelligent execution plan based on query analysis"""
        logger.info("Creating execution plan...")
        
        try:
            query_analysis = state["analysis_plan"].get("query_analysis", {})
            complexity = query_analysis.get("complexity", "moderate")
            required_data_types = query_analysis.get("required_data_types", [])
            
            # Determine required agents
            required_agents = []
            agent_dependencies = {}
            
            # Smart agent selection based on query analysis
            if any("user" in data_type.lower() for data_type in required_data_types):
                if "user_analysis" in self.specialized_agents:
                    required_agents.append("user_analysis")
            
            if any("transaction" in data_type.lower() for data_type in required_data_types):
                if "transaction_analysis" in self.specialized_agents:
                    required_agents.append("transaction_analysis")
                    # Transactions often depend on user data
                    if "user_analysis" in required_agents:
                        agent_dependencies["transaction_analysis"] = ["user_analysis"]
            
            # Always include analytics for complex queries
            if complexity in ["moderate", "complex"] and "analytics" in self.specialized_agents:
                required_agents.append("analytics")
                agent_dependencies["analytics"] = [agent for agent in required_agents if agent != "analytics"]
            
            # Determine execution strategy
            if len(required_agents) <= 1:
                execution_strategy = "single_agent"
            elif agent_dependencies:
                execution_strategy = "sequential_with_dependencies"
            else:
                execution_strategy = "parallel"
            
            state["required_agents"] = required_agents
            state["agent_dependencies"] = agent_dependencies
            state["execution_strategy"] = execution_strategy
            state["analysis_plan"]["execution_plan"] = {
                "strategy": execution_strategy,
                "agents": required_agents,
                "dependencies": agent_dependencies,
                "estimated_complexity": complexity
            }
            state["current_step"] = "plan_created"
            
            logger.info(f"Execution plan: {execution_strategy} with agents: {required_agents}")
            
        except Exception as e:
            logger.error(f"Execution planning failed: {e}")
            state["errors"].append(f"Execution planning failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def execute_agents(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Execute specialized agents according to plan"""
        logger.info("Executing specialized agents...")
        
        try:
            required_agents = state["required_agents"]
            agent_dependencies = state["agent_dependencies"]
            execution_strategy = state["execution_strategy"]
            
            agent_results = {}
            context = {}
            
            if execution_strategy == "parallel":
                # Execute all agents in parallel
                tasks = []
                for agent_name in required_agents:
                    if agent_name in self.specialized_agents:
                        agent = self.specialized_agents[agent_name]
                        task = agent.analyze(state["query"], context)
                        tasks.append((agent_name, task))
                
                if tasks:
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    for (agent_name, _), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            agent_results[agent_name] = {"success": False, "error": str(result)}
                        else:
                            agent_results[agent_name] = result
            
            elif execution_strategy in ["sequential_with_dependencies", "single_agent"]:
                # Execute agents in dependency order
                executed_agents = set()
                
                while len(executed_agents) < len(required_agents):
                    ready_agents = []
                    
                    for agent_name in required_agents:
                        if agent_name not in executed_agents:
                            dependencies = agent_dependencies.get(agent_name, [])
                            if all(dep in executed_agents for dep in dependencies):
                                ready_agents.append(agent_name)
                    
                    if not ready_agents:
                        # Circular dependency or other issue
                        remaining = set(required_agents) - executed_agents
                        state["warnings"].append(f"Circular dependency detected, executing remaining agents: {remaining}")
                        ready_agents = list(remaining)
                    
                    # Execute ready agents
                    for agent_name in ready_agents:
                        if agent_name in self.specialized_agents:
                            agent = self.specialized_agents[agent_name]
                            try:
                                result = await agent.analyze(state["query"], context)
                                agent_results[agent_name] = result
                                
                                # Update context for dependent agents
                                if result.get("success") and result.get("data"):
                                    context[agent_name] = result["data"]
                                    # Special handling for user IDs
                                    if agent_name == "user_analysis" and "user_ids" in result.get("metadata", {}):
                                        context["user_ids"] = result["metadata"]["user_ids"]
                                        
                            except Exception as e:
                                agent_results[agent_name] = {"success": False, "error": str(e)}
                                logger.error(f"Agent {agent_name} failed: {e}")
                        
                        executed_agents.add(agent_name)
            
            state["agent_results"] = agent_results
            state["current_step"] = "agents_executed"
            
            # Log execution summary
            successful_agents = [name for name, result in agent_results.items() if result.get("success")]
            failed_agents = [name for name, result in agent_results.items() if not result.get("success")]
            
            logger.info(f"Agent execution complete. Successful: {successful_agents}, Failed: {failed_agents}")
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            state["errors"].append(f"Agent execution failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def correlate_data(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Correlate data from different agents to find relationships"""
        logger.info("Correlating data across agents...")
        
        try:
            agent_results = state["agent_results"]
            correlations = []
            consolidated_data = {"records": [], "metadata": {}}
            
            # Extract all successful data
            all_data = {}
            for agent_name, result in agent_results.items():
                if result.get("success") and result.get("data"):
                    all_data[agent_name] = result["data"]
                    consolidated_data["records"].extend(result["data"])
            
            # Find correlations between datasets
            if len(all_data) > 1:
                # Example: Correlate users and transactions
                if "user_analysis" in all_data and "transaction_analysis" in all_data:
                    user_data = all_data["user_analysis"]
                    transaction_data = all_data["transaction_analysis"]
                    
                    # Find common user IDs
                    user_ids = set()
                    transaction_user_ids = set()
                    
                    for record in user_data:
                        if "user_id" in record:
                            user_ids.add(record["user_id"])
                    
                    for record in transaction_data:
                        if "user_id" in record:
                            transaction_user_ids.add(record["user_id"])
                    
                    common_users = user_ids.intersection(transaction_user_ids)
                    
                    if common_users:
                        correlations.append({
                            "type": "user_transaction_correlation",
                            "description": f"Found {len(common_users)} users with both profile and transaction data",
                            "common_user_count": len(common_users),
                            "user_coverage": len(common_users) / max(len(user_ids), 1) * 100,
                            "transaction_coverage": len(common_users) / max(len(transaction_user_ids), 1) * 100
                        })
            
            # Update consolidated metadata
            consolidated_data["metadata"] = {
                "total_records": len(consolidated_data["records"]),
                "data_sources": list(all_data.keys()),
                "correlations_found": len(correlations),
                "correlation_types": [corr["type"] for corr in correlations]
            }
            
            state["data_correlations"] = correlations
            state["consolidated_data"] = consolidated_data
            state["current_step"] = "data_correlated"
            
            logger.info(f"Data correlation complete. Found {len(correlations)} correlations")
            
        except Exception as e:
            logger.error(f"Data correlation failed: {e}")
            state["errors"].append(f"Data correlation failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def generate_insights(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Generate AI-powered insights from consolidated data"""
        logger.info("Generating AI insights...")
        
        if not state["user_preferences"].get("include_insights", True):
            state["insights"] = "Insights generation skipped by user preference"
            state["confidence_score"] = 0.0
            state["current_step"] = "insights_generated"
            return state
        
        try:
            consolidated_data = state["consolidated_data"]
            correlations = state["data_correlations"]
            
            # Prepare data summary for LLM
            data_summary = {
                "total_records": consolidated_data["metadata"].get("total_records", 0),
                "data_sources": consolidated_data["metadata"].get("data_sources", []),
                "correlations": [corr["description"] for corr in correlations],
                "sample_data": consolidated_data["records"][:5]  # First 5 records as sample
            }
            
            insights_prompt = f"""
            Analyze the following multi-database query results and provide comprehensive insights:
            
            Original Query: "{state['query']}"
            
            Data Summary:
            {json.dumps(data_summary, indent=2, default=str)}
            
            Data Correlations Found:
            {json.dumps(correlations, indent=2)}
            
            Please provide:
            1. Key findings and patterns
            2. Data quality observations
            3. Business insights and implications
            4. Statistical summaries where relevant
            5. Confidence level in the analysis (0-100%)
            
            Format your response as structured insights with clear sections.
            """
            
            llm_response = await self.llm_client.generate_completion(
                insights_prompt,
                max_tokens=1000,
                temperature=0.3,
                system_message="You are an expert data analyst providing actionable insights from multi-database analysis."
            )
            
            if llm_response.success:
                state["insights"] = llm_response.content
                # Extract confidence score if mentioned (simple approach)
                confidence_score = 85.0  # Default confidence
                if "confidence" in llm_response.content.lower():
                    import re
                    confidence_matches = re.findall(r'(\d+)%', llm_response.content)
                    if confidence_matches:
                        confidence_score = float(confidence_matches[-1])  # Use last mentioned percentage
                
                state["confidence_score"] = min(confidence_score, 100.0)
            else:
                # Generate basic insights without LLM
                basic_insights = self._generate_basic_insights(state)
                state["insights"] = basic_insights
                state["confidence_score"] = 60.0  # Lower confidence for basic insights
                state["warnings"].append("LLM unavailable, generated basic insights")
            
            state["current_step"] = "insights_generated"
            
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            state["errors"].append(f"Insights generation failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def create_recommendations(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Create actionable recommendations based on insights"""
        logger.info("Creating recommendations...")
        
        try:
            consolidated_data = state["consolidated_data"]
            insights = state["insights"]
            correlations = state["data_correlations"]
            
            recommendations = []
            
            # Data-driven recommendations
            total_records = consolidated_data["metadata"].get("total_records", 0)
            if total_records == 0:
                recommendations.append("No data found - verify database connections and query parameters")
            elif total_records < 10:
                recommendations.append("Limited data available - consider expanding query scope or date range")
            
            # Correlation-based recommendations
            if len(correlations) == 0:
                recommendations.append("No data correlations found - consider running individual database analyses")
            else:
                recommendations.append(f"Strong data correlations found ({len(correlations)}) - suitable for cross-database analysis")
            
            # Query-specific recommendations
            query_lower = state["query"].lower()
            if "user" in query_lower and "transaction" in query_lower:
                recommendations.append("For user-transaction analysis, consider filtering by date range for better performance")
            
            if "analyze" in query_lower or "insight" in query_lower:
                recommendations.append("Run this analysis periodically to track trends over time")
            
            # Performance recommendations
            execution_time = state.get("execution_time", 0)
            if execution_time > 30:
                recommendations.append("Query took longer than expected - consider optimizing with more specific filters")
            
            # Add LLM-generated recommendations if available
            if state["confidence_score"] > 70:
                recommendations.append("High confidence analysis - results are suitable for decision making")
            else:
                recommendations.append("Moderate confidence analysis - consider additional data validation")
            
            state["recommendations"] = recommendations
            state["current_step"] = "recommendations_created"
            
        except Exception as e:
            logger.error(f"Recommendation creation failed: {e}")
            state["errors"].append(f"Recommendation creation failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    async def finalize_results(self, state: EnhancedMultiAgentState) -> EnhancedMultiAgentState:
        """Finalize and package results"""
        logger.info("Finalizing results...")
        
        try:
            # Update metadata with final information
            state["metadata"]["end_time"] = datetime.now().isoformat()
            state["metadata"]["processing_steps"] = [
                "query_analyzed", "plan_created", "agents_executed", 
                "data_correlated", "insights_generated", "recommendations_created"
            ]
            state["metadata"]["final_record_count"] = state["consolidated_data"]["metadata"].get("total_records", 0)
            state["metadata"]["confidence_score"] = state["confidence_score"]
            state["metadata"]["recommendation_count"] = len(state["recommendations"])
            
            state["current_step"] = "complete"
            
        except Exception as e:
            logger.error(f"Result finalization failed: {e}")
            state["errors"].append(f"Result finalization failed: {e}")
            state["current_step"] = "error"
        
        return state
    
    def _generate_basic_insights(self, state: EnhancedMultiAgentState) -> str:
        """Generate basic insights without LLM"""
        consolidated_data = state["consolidated_data"]
        correlations = state["data_correlations"]
        
        insights = f"""
# Analysis Results for: "{state['query']}"

## Data Summary
- Total records retrieved: {consolidated_data['metadata'].get('total_records', 0)}
- Data sources: {', '.join(consolidated_data['metadata'].get('data_sources', []))}
- Correlations found: {len(correlations)}

## Key Findings
"""
        
        for i, correlation in enumerate(correlations, 1):
            insights += f"{i}. {correlation['description']}\n"
        
        insights += f"""
## System Performance
- Execution strategy: {state['execution_strategy']}
- Agents used: {', '.join(state['required_agents'])}
- Processing completed successfully

## Next Steps
The data has been successfully retrieved and correlated across multiple databases.
Consider running this analysis periodically to track trends over time.
        """
        
        return insights
    
    # Additional orchestrator methods for MCP integration
    async def get_database_schemas(self, database_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema information"""
        if not self.db_manager:
            return {"error": "Database manager not initialized"}
        
        return await self.db_manager.get_all_schemas(database_name)
    
    async def test_all_connections(self, include_details: bool = False) -> Dict[str, Any]:
        """Test all database connections"""
        if not self.db_manager:
            return {"error": "Database manager not initialized"}
        
        return await self.db_manager.test_all_connections(include_details)
    
    async def explain_query_plan(self, query: str) -> Dict[str, Any]:
        """Explain how a query would be executed"""
        # Create a mock state to run through planning
        mock_state: EnhancedMultiAgentState = {
            "query": query,
            "session_id": "explanation",
            "user_preferences": {},
            "analysis_plan": {},
            "execution_strategy": "",
            "required_agents": [],
            "agent_dependencies": {},
            "agent_results": {},
            "consolidated_data": {},
            "data_correlations": [],
            "insights": "",
            "confidence_score": 0.0,
            "recommendations": [],
            "current_step": "starting",
            "execution_time": 0.0,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        # Run analysis and planning phases
        analyzed_state = await self.analyze_query(mock_state)
        planned_state = await self.create_execution_plan(analyzed_state)
        
        return {
            "query": query,
            "analysis": planned_state["analysis_plan"],
            "execution_strategy": planned_state["execution_strategy"],
            "required_agents": planned_state["required_agents"],
            "agent_dependencies": planned_state["agent_dependencies"],
            "estimated_complexity": planned_state["analysis_plan"].get("query_analysis", {}).get("complexity", "unknown")
        }
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        if self.db_manager:
            await self.db_manager.cleanup()
        
        for agent in self.specialized_agents.values():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        logger.info("Enhanced Orchestrator cleanup completed")