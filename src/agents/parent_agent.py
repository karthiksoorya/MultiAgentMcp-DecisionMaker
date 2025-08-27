# Parent Agent - Main orchestrator for multi-agent system
import asyncio
import json
import litellm
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
import logging

from utils.config import Config
from utils.mcp_client import DatabaseAgent

logger = logging.getLogger(__name__)

# State definition for LangGraph
class MultiAgentState(TypedDict):
    user_query: str
    analysis_plan: Dict[str, Any]
    agent_responses: List[Dict[str, Any]]
    consolidated_data: Dict[str, Any]
    final_insights: str
    current_step: str
    errors: List[str]

class ParentAgent:
    """Main orchestrator agent"""
    
    def __init__(self):
        self.config = Config()
        self.database_agents = {}
        self.llm_client = self._setup_llm()
        
    def _setup_llm(self):
        """Setup LiteLLM client"""
        if not self.config.LLM.api_key:
            logger.warning("No OpenRouter API key found")
            return None
            
        # Configure LiteLLM for OpenRouter
        litellm.api_key = self.config.LLM.api_key
        litellm.api_base = self.config.LLM.base_url
        return litellm
    
    async def initialize(self) -> bool:
        """Initialize parent agent and all sub-agents"""
        logger.info("Initializing Parent Agent and sub-agents...")
        
        if not self.config.DATABASES:
            logger.error("No databases configured")
            return False
        
        # Create and initialize database agents
        for db_name, db_config in self.config.DATABASES.items():
            agent = DatabaseAgent(db_name, db_config, self.llm_client)
            
            if await agent.initialize():
                self.database_agents[db_name] = agent
                logger.info(f"Initialized agent for {db_name}")
            else:
                logger.error(f"Failed to initialize agent for {db_name}")
        
        if not self.database_agents:
            logger.error("No database agents initialized successfully")
            return False
        
        logger.info(f"Parent Agent initialized with {len(self.database_agents)} sub-agents")
        return True
    
    async def cleanup(self):
        """Cleanup all agents"""
        for agent in self.database_agents.values():
            await agent.cleanup()
        logger.info("Parent Agent cleanup completed")
    
    async def understand_query(self, state: MultiAgentState) -> MultiAgentState:
        """Analyze user query and create execution plan"""
        logger.info(f"Analyzing query: {state['user_query']}")
        
        try:
            # Get capabilities from all agents
            agent_capabilities = {}
            for agent_name, agent in self.database_agents.items():
                capability = await agent.analyze_query_capability(state['user_query'])
                agent_capabilities[agent_name] = capability
            
            # Create execution plan
            plan = await self._create_execution_plan(state['user_query'], agent_capabilities)
            
            state['analysis_plan'] = plan
            state['current_step'] = 'planning_complete'
            logger.info("Query analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state['errors'].append(f"Query analysis failed: {e}")
            state['current_step'] = 'error'
        
        return state
    
    async def _create_execution_plan(self, user_query: str, capabilities: Dict) -> Dict[str, Any]:
        """Create execution plan using LLM"""
        
        if not self.llm_client:
            # Fallback plan without LLM
            return {
                "execution_order": list(capabilities.keys()),
                "dependencies": {},
                "reasoning": "Basic sequential execution (LLM not available)"
            }
        
        try:
            prompt = f"""
            Analyze this user query and create an execution plan for multi-agent database analysis:
            
            Query: "{user_query}"
            
            Available agents and their capabilities:
            {json.dumps(capabilities, indent=2)}
            
            Create an execution plan that:
            1. Determines which agents are needed
            2. Identifies the optimal execution order
            3. Identifies data dependencies between agents
            
            Respond ONLY in JSON format:
            {{
                "execution_order": ["agent1", "agent2"],
                "dependencies": {{"agent2": "needs user_ids from agent1"}},
                "reasoning": "explanation of the plan"
            }}
            """
            
            response = litellm.completion(
                model=self.config.LLM.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                plan = json.loads(plan_text)
                return plan
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("LLM response was not valid JSON, using fallback plan")
                return {
                    "execution_order": list(capabilities.keys()),
                    "dependencies": {},
                    "reasoning": "Fallback plan due to JSON parsing error"
                }
                
        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            # Fallback plan
            return {
                "execution_order": list(capabilities.keys()),
                "dependencies": {},
                "reasoning": f"Fallback plan due to LLM error: {e}"
            }
    
    async def execute_agents(self, state: MultiAgentState) -> MultiAgentState:
        """Execute database agents according to plan"""
        logger.info("Executing database agents...")
        
        try:
            plan = state['analysis_plan']
            execution_order = plan.get('execution_order', [])
            dependencies = plan.get('dependencies', {})
            
            agent_responses = []
            context = {}
            
            # Execute agents in planned order
            for agent_name in execution_order:
                if agent_name not in self.database_agents:
                    logger.warning(f"Agent {agent_name} not found, skipping")
                    continue
                
                logger.info(f"Executing agent: {agent_name}")
                agent = self.database_agents[agent_name]
                
                # Check if this agent needs data from previous agents
                if agent_name in dependencies:
                    # Extract required data from previous responses
                    context = self._extract_context_for_agent(agent_name, agent_responses, dependencies)
                
                # Execute the agent
                response = await agent.execute_analysis(state['user_query'], context)
                agent_responses.append(response)
                
                # Log execution result
                if response.get('success'):
                    logger.info(f"Agent {agent_name} executed successfully")
                else:
                    logger.warning(f"Agent {agent_name} failed: {response.get('error')}")
            
            state['agent_responses'] = agent_responses
            state['current_step'] = 'agent_execution_complete'
            logger.info("Agent execution completed")
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            state['errors'].append(f"Agent execution failed: {e}")
            state['current_step'] = 'error'
        
        return state
    
    def _extract_context_for_agent(self, agent_name: str, previous_responses: List[Dict], dependencies: Dict) -> Dict[str, Any]:
        """Extract context data needed by an agent from previous responses"""
        context = {}
        
        # Look for user_ids in previous responses (common dependency)
        for response in previous_responses:
            if response.get('success') and 'metadata' in response:
                metadata = response['metadata']
                if 'user_ids' in metadata:
                    context['user_ids'] = metadata['user_ids']
                    break
        
        return context
    
    async def consolidate_data(self, state: MultiAgentState) -> MultiAgentState:
        """Consolidate data from all agent responses"""
        logger.info("Consolidating data from agent responses...")
        
        try:
            consolidated = {
                'successful_agents': [],
                'failed_agents': [],
                'total_records': 0,
                'data_by_agent': {}
            }
            
            for response in state['agent_responses']:
                agent_name = response.get('agent_name', 'unknown')
                
                if response.get('success'):
                    consolidated['successful_agents'].append(agent_name)
                    consolidated['data_by_agent'][agent_name] = response.get('data', [])
                    consolidated['total_records'] += len(response.get('data', []))
                else:
                    consolidated['failed_agents'].append(agent_name)
            
            state['consolidated_data'] = consolidated
            state['current_step'] = 'consolidation_complete'
            logger.info(f"Data consolidation completed: {consolidated['total_records']} total records")
            
        except Exception as e:
            logger.error(f"Data consolidation failed: {e}")
            state['errors'].append(f"Data consolidation failed: {e}")
            state['current_step'] = 'error'
        
        return state
    
    async def generate_insights(self, state: MultiAgentState) -> MultiAgentState:
        """Generate final insights from consolidated data"""
        logger.info("Generating insights from consolidated data...")
        
        try:
            consolidated = state['consolidated_data']
            
            if not self.llm_client:
                # Generate basic insights without LLM
                insights = self._generate_basic_insights(state['user_query'], consolidated)
            else:
                # Generate advanced insights with LLM
                insights = await self._generate_llm_insights(state['user_query'], consolidated)
            
            state['final_insights'] = insights
            state['current_step'] = 'complete'
            logger.info("Insights generation completed")
            
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            state['errors'].append(f"Insights generation failed: {e}")
            state['current_step'] = 'error'
        
        return state
    
    def _generate_basic_insights(self, user_query: str, consolidated_data: Dict) -> str:
        """Generate basic insights without LLM"""
        insights = f"""
# Analysis Results for: "{user_query}"

## Data Summary
- Successful agents: {len(consolidated_data['successful_agents'])}
- Failed agents: {len(consolidated_data['failed_agents'])}
- Total records retrieved: {consolidated_data['total_records']}

## Agent Results
"""
        
        for agent_name in consolidated_data['successful_agents']:
            data = consolidated_data['data_by_agent'][agent_name]
            insights += f"- {agent_name}: {len(data)} records\n"
        
        if consolidated_data['failed_agents']:
            insights += f"\n## Failed Agents\n"
            for agent_name in consolidated_data['failed_agents']:
                insights += f"- {agent_name}: Failed to retrieve data\n"
        
        insights += f"\n## Next Steps\nThe data has been successfully retrieved from multiple databases and is ready for further analysis."
        
        return insights
    
    async def _generate_llm_insights(self, user_query: str, consolidated_data: Dict) -> str:
        """Generate advanced insights using LLM"""
        
        try:
            # Create a summary of the data for the LLM
            data_summary = {
                "query": user_query,
                "successful_agents": consolidated_data['successful_agents'],
                "total_records": consolidated_data['total_records'],
                "sample_data": {}
            }
            
            # Add sample data from each agent (limited for token efficiency)
            for agent_name, data in consolidated_data['data_by_agent'].items():
                if data:
                    # Take first few records as sample
                    data_summary["sample_data"][agent_name] = data[:3]
            
            prompt = f"""
            Analyze the following multi-database query results and provide actionable insights:
            
            Original Query: "{user_query}"
            
            Data Summary:
            {json.dumps(data_summary, indent=2, default=str)}
            
            Please provide:
            1. Key findings from the data
            2. Patterns or trends identified
            3. Actionable recommendations
            4. Any data quality observations
            
            Format your response in markdown with clear sections.
            """
            
            response = litellm.completion(
                model=self.config.LLM.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM insights generation failed: {e}")
            return self._generate_basic_insights(user_query, consolidated_data)

def create_workflow(parent_agent: ParentAgent) -> StateGraph:
    """Create LangGraph workflow"""
    
    workflow = StateGraph(MultiAgentState)
    
    # Add nodes
    workflow.add_node("understand_query", parent_agent.understand_query)
    workflow.add_node("execute_agents", parent_agent.execute_agents)
    workflow.add_node("consolidate_data", parent_agent.consolidate_data)
    workflow.add_node("generate_insights", parent_agent.generate_insights)
    
    # Add edges
    workflow.add_edge("understand_query", "execute_agents")
    workflow.add_edge("execute_agents", "consolidate_data")
    workflow.add_edge("consolidate_data", "generate_insights")
    workflow.add_edge("generate_insights", END)
    
    # Set entry point
    workflow.set_entry_point("understand_query")
    
    return workflow.compile()