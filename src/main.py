# Main application entry point for Multi-Agent PostgreSQL Analysis System
import asyncio
import logging
import sys
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.parent_agent import ParentAgent, create_workflow, MultiAgentState
from utils.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiAgentSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.parent_agent = ParentAgent()
        self.workflow = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the entire system"""
        logger.info("Initializing Multi-Agent System...")
        
        try:
            # Initialize parent agent and sub-agents
            if not await self.parent_agent.initialize():
                logger.error("Failed to initialize parent agent")
                return False
            
            # Create LangGraph workflow
            self.workflow = create_workflow(self.parent_agent)
            
            self.initialized = True
            logger.info("Multi-Agent System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a user query through the multi-agent system"""
        
        if not self.initialized:
            return {
                "success": False,
                "error": "System not initialized",
                "insights": ""
            }
        
        logger.info(f"Processing query: {user_query}")
        
        # Create initial state
        initial_state: MultiAgentState = {
            "user_query": user_query,
            "analysis_plan": {},
            "agent_responses": [],
            "consolidated_data": {},
            "final_insights": "",
            "current_step": "starting",
            "errors": []
        }
        
        try:
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            
            if result["current_step"] == "complete":
                logger.info("Query processed successfully")
                return {
                    "success": True,
                    "insights": result["final_insights"],
                    "data_summary": result["consolidated_data"],
                    "execution_plan": result["analysis_plan"]
                }
            else:
                logger.error(f"Query processing failed at step: {result['current_step']}")
                return {
                    "success": False,
                    "error": f"Processing failed at step: {result['current_step']}",
                    "errors": result.get("errors", []),
                    "insights": result.get("final_insights", "")
                }
                
        except Exception as e:
            logger.error(f"Query processing crashed: {e}")
            return {
                "success": False,
                "error": f"System error: {e}",
                "insights": ""
            }
    
    async def cleanup(self):
        """Cleanup system resources"""
        if self.parent_agent:
            await self.parent_agent.cleanup()
        logger.info("System cleanup completed")

async def interactive_mode(system: MultiAgentSystem):
    """Run system in interactive mode"""
    print("\nMulti-Agent PostgreSQL Analysis System")
    print("=" * 50)
    print("Enter your queries below. Type 'quit' to exit.\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nProcessing query...")
            result = await system.process_query(query)
            
            if result["success"]:
                print("\nResults:")
                print("-" * 40)
                print(result["insights"])
                
                if "data_summary" in result:
                    data_summary = result["data_summary"]
                    print(f"\nData Summary:")
                    print(f"   Total records: {data_summary.get('total_records', 0)}")
                    print(f"   Successful agents: {len(data_summary.get('successful_agents', []))}")
                    
                    if data_summary.get('failed_agents'):
                        print(f"   Failed agents: {', '.join(data_summary['failed_agents'])}")
            else:
                print("\nQuery failed:")
                print(f"   Error: {result['error']}")
                if result.get("errors"):
                    print("   Details:")
                    for error in result["errors"]:
                        print(f"     - {error}")
            
            print("\n" + "="*50)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")

async def single_query_mode(system: MultiAgentSystem, query: str):
    """Process a single query"""
    print(f"Processing query: {query}")
    
    result = await system.process_query(query)
    
    if result["success"]:
        print("\nQuery processed successfully!")
        print("=" * 50)
        print(result["insights"])
        return True
    else:
        print(f"\nQuery failed: {result['error']}")
        if result.get("errors"):
            print("Errors:")
            for error in result["errors"]:
                print(f"  - {error}")
        return False

def print_system_info():
    """Print system information"""
    config = Config()
    
    print("\nSystem Configuration:")
    print(f"   Databases configured: {len(config.DATABASES)}")
    for db_name, db_config in config.DATABASES.items():
        print(f"   - {db_name}: {db_config.host}/{db_config.database}")
    
    print(f"   LLM configured: {'Yes' if config.LLM.api_key else 'No'}")
    print(f"   Debug mode: {config.DEBUG}")

async def main():
    """Main application entry point"""
    
    # Print system info
    print_system_info()
    
    # Initialize system
    system = MultiAgentSystem()
    
    try:
        if not await system.initialize():
            print("Failed to initialize system. Check your configuration.")
            sys.exit(1)
        
        # Check command line arguments
        if len(sys.argv) > 1:
            # Single query mode
            query = " ".join(sys.argv[1:])
            success = await single_query_mode(system, query)
            sys.exit(0 if success else 1)
        else:
            # Interactive mode
            await interactive_mode(system)
    
    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())