# Setup verification and basic tests
import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from utils.config import Config
from utils.mcp_client import DatabaseAgent
from agents.parent_agent import ParentAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SetupTester:
    """Test system setup and configuration"""
    
    def __init__(self):
        load_dotenv()
        self.config = Config()
    
    async def test_dependencies(self) -> bool:
        """Test that all required dependencies are installed"""
        print("Testing Dependencies...")
        
        try:
            import langgraph
            import litellm
            import asyncpg
            import pandas
            import pydantic
            print("All dependencies imported successfully")
            return True
        except ImportError as e:
            print(f"Dependency import failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration loading"""
        print("Testing Configuration...")
        
        if not self.config.DATABASES:
            print("No databases configured")
            return False
        
        print(f"Found {len(self.config.DATABASES)} database(s)")
        
        for db_name, db_config in self.config.DATABASES.items():
            print(f"   - {db_name}: {db_config.host}")
        
        if self.config.LLM.api_key:
            print("LLM API key configured")
        else:
            print("No LLM API key (basic insights only)")
        
        return True
    
    async def test_database_connections(self) -> bool:
        """Test database connections"""
        print("Testing Database Connections...")
        
        success_count = 0
        
        for db_name, db_config in self.config.DATABASES.items():
            try:
                agent = DatabaseAgent(db_name, db_config)
                
                if await agent.initialize():
                    tables = await agent.mcp_client.list_tables()
                    print(f"{db_name}: Connected, found {len(tables)} tables")
                    
                    if tables:
                        # Test sample query
                        sample_data = await agent.mcp_client.get_sample_data(tables[0], limit=1)
                        print(f"   Sample from {tables[0]}: {len(sample_data)} record(s)")
                    
                    await agent.cleanup()
                    success_count += 1
                else:
                    print(f"{db_name}: Connection failed")
                    
            except Exception as e:
                print(f"{db_name}: Error - {e}")
        
        return success_count == len(self.config.DATABASES)
    
    async def test_simple_query(self) -> bool:
        """Test a simple query end-to-end"""
        print("Testing Simple Query...")
        
        try:
            from main import MultiAgentSystem
            
            system = MultiAgentSystem()
            
            if await system.initialize():
                # Test with a simple query
                result = await system.process_query("Show me a summary of the available data")
                
                if result["success"]:
                    print("Simple query test passed")
                    print(f"   Insight preview: {result['insights'][:100]}...")
                    await system.cleanup()
                    return True
                else:
                    print(f"Query failed: {result['error']}")
                    await system.cleanup()
                    return False
            else:
                print("System initialization failed")
                return False
                
        except Exception as e:
            print(f"Simple query test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all setup tests"""
        print("Multi-Agent System Setup Tests")
        print("=" * 50)
        
        tests = [
            ("Dependencies", self.test_dependencies()),
            ("Configuration", self.test_configuration()),
            ("Database Connections", self.test_database_connections()),
            ("Simple Query", self.test_simple_query())
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n--- {test_name} ---")
            
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
                
            results.append((test_name, result))
        
        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print("\n" + "=" * 50)
        print(f"TEST SUMMARY: {passed}/{total} tests passed")
        
        if passed == total:
            print("All tests passed! System is ready to use.")
            print("\nYou can now run:")
            print("  python src/main.py                    # Interactive mode")
            print("  python src/main.py 'your query here'  # Single query")
        else:
            print("Some tests failed. Please fix the issues before using the system.")
            failed_tests = [name for name, result in results if not result]
            print(f"Failed tests: {', '.join(failed_tests)}")
        
        return passed == total

if __name__ == "__main__":
    tester = SetupTester()
    asyncio.run(tester.run_all_tests())