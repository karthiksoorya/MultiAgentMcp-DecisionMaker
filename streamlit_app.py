import streamlit as st
import asyncio
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our multi-agent system
from main import MultiAgentSystem
from utils.config import Config
from utils.mcp_client import DatabaseAgent

# Page config
st.set_page_config(
    page_title="Multi-Agent PostgreSQL Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.config = Config()

async def initialize_system():
    """Initialize the multi-agent system"""
    if st.session_state.system is None:
        st.session_state.system = MultiAgentSystem()
    
    if not st.session_state.initialized:
        with st.spinner("Initializing Multi-Agent System..."):
            success = await st.session_state.system.initialize()
            st.session_state.initialized = success
            return success
    return True

async def process_query_async(query):
    """Process query asynchronously"""
    if not st.session_state.initialized:
        await initialize_system()
    
    if st.session_state.system:
        result = await st.session_state.system.process_query(query)
        # Add to history
        st.session_state.query_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'result': result
        })
        return result
    return {"success": False, "error": "System not initialized"}

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Agent PostgreSQL Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["🏠 Dashboard", "🔍 Query Interface", "⚙️ Configuration", "📊 Analytics", "📋 History"],
            icons=['house', 'search', 'gear', 'graph-up', 'clock-history'],
            menu_icon="cast",
            default_index=0,
        )
    
    if selected == "🏠 Dashboard":
        show_dashboard()
    elif selected == "🔍 Query Interface":
        show_query_interface()
    elif selected == "⚙️ Configuration":
        show_configuration()
    elif selected == "📊 Analytics":
        show_analytics()
    elif selected == "📋 History":
        show_history()

def show_dashboard():
    """Show system dashboard"""
    st.header("🏠 System Dashboard")
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        config = st.session_state.config
        db_count = len(config.DATABASES)
        st.metric("Databases Configured", db_count, delta=None)
    
    with col2:
        llm_status = "✅ Connected" if config.LLM.api_key else "❌ Not Configured"
        st.metric("LLM Status", llm_status)
    
    with col3:
        queries_count = len(st.session_state.query_history)
        st.metric("Queries Executed", queries_count)
    
    with col4:
        system_status = "🟢 Ready" if st.session_state.initialized else "🔴 Not Initialized"
        st.metric("System Status", system_status)
    
    # Database connections status
    st.subheader("📡 Database Connections")
    
    if st.button("🔄 Test Connections", type="primary"):
        test_connections()
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    if st.session_state.query_history:
        recent_queries = st.session_state.query_history[-5:]  # Last 5 queries
        for i, query_info in enumerate(reversed(recent_queries)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {query_info['query'][:50]}..."):
                st.write(f"**Timestamp:** {query_info['timestamp']}")
                st.write(f"**Query:** {query_info['query']}")
                st.write(f"**Success:** {'✅' if query_info['result']['success'] else '❌'}")
                if query_info['result']['success']:
                    st.write("**Insights:**")
                    st.write(query_info['result']['insights'])
    else:
        st.info("No queries executed yet. Go to Query Interface to start!")

def show_query_interface():
    """Show query interface"""
    st.header("🔍 Query Interface")
    
    # Initialize system if needed
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing system..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(initialize_system())
                loop.close()
                
                if success:
                    st.success("✅ System initialized successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize system. Check your database configuration.")
                    return
    
    # Query input
    st.subheader("💬 Enter Your Query")
    
    # Predefined example queries
    example_queries = [
        "Show me all active users",
        "Find high-value customers and their recent transactions", 
        "Analyze user behavior patterns across databases",
        "Give me insights about database structure",
        "Compare transaction volumes by user segment"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Query:",
            placeholder="Enter your natural language query here...",
            height=100
        )
    
    with col2:
        st.write("**Example Queries:**")
        for example in example_queries:
            if st.button(f"💡 {example[:30]}...", key=f"example_{example}"):
                st.session_state.example_query = example
                st.rerun()
    
    # Use example query if selected
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    
    # Execute query
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Execute Query", type="primary", disabled=not query or not st.session_state.initialized):
            execute_query(query)

def execute_query(query):
    """Execute the query and display results"""
    start_time = time.time()
    
    with st.spinner(f"🤖 Processing query: {query[:50]}..."):
        # Run async query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_query_async(query))
        loop.close()
    
    execution_time = time.time() - start_time
    
    # Display results
    st.subheader("📊 Query Results")
    
    if result['success']:
        # Success message
        st.markdown(
            f'<div class="success-message">✅ Query executed successfully in {execution_time:.2f} seconds</div>',
            unsafe_allow_html=True
        )
        
        # Display insights
        st.subheader("🧠 AI-Generated Insights")
        st.markdown(result['insights'])
        
        # Display data summary if available
        if 'data_summary' in result:
            data_summary = result['data_summary']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", data_summary.get('total_records', 0))
            with col2:
                st.metric("Successful Agents", len(data_summary.get('successful_agents', [])))
            with col3:
                failed_count = len(data_summary.get('failed_agents', []))
                st.metric("Failed Agents", failed_count, delta=-failed_count if failed_count > 0 else None)
            
            # Agent details
            if data_summary.get('successful_agents'):
                st.subheader("🤖 Agent Execution Details")
                for agent_name in data_summary['successful_agents']:
                    agent_data = data_summary['data_by_agent'].get(agent_name, [])
                    with st.expander(f"Agent: {agent_name} ({len(agent_data)} records)"):
                        if agent_data:
                            df = pd.DataFrame(agent_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No data returned by this agent")
        
        # Display execution plan
        if 'execution_plan' in result:
            with st.expander("🔧 Execution Plan Details"):
                st.json(result['execution_plan'])
    
    else:
        # Error message
        st.markdown(
            f'<div class="error-message">❌ Query failed: {result["error"]}</div>',
            unsafe_allow_html=True
        )
        
        if result.get('errors'):
            st.subheader("🔍 Error Details")
            for error in result['errors']:
                st.error(f"• {error}")

def show_configuration():
    """Show configuration interface"""
    st.header("⚙️ System Configuration")
    
    config = st.session_state.config
    
    # LLM Configuration
    st.subheader("🧠 LLM Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        api_key_status = "✅ Configured" if config.LLM.api_key else "❌ Not Set"
        st.metric("OpenRouter API Key", api_key_status)
        st.info(f"Model: {config.LLM.model}")
    
    with col2:
        if st.button("🔄 Update LLM Settings"):
            st.info("To update LLM settings, edit the .env file and restart the application")
    
    # Database Configuration
    st.subheader("🗄️ Database Configuration")
    
    for db_name, db_config in config.DATABASES.items():
        with st.expander(f"Database: {db_name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Host:** {db_config.host}")
                st.write(f"**Database:** {db_config.database}")
                st.write(f"**Port:** {db_config.port}")
            
            with col2:
                st.write(f"**Username:** {db_config.username}")
                st.write(f"**SSL:** {'Enabled' if db_config.ssl else 'Disabled'}")
                
                # Test connection button
                if st.button(f"Test Connection", key=f"test_{db_name}"):
                    test_single_connection(db_name, db_config)
    
    # Add new database section
    st.subheader("➕ Add New Database")
    with st.expander("Configure New Database"):
        st.info("To add a new database, update the .env file with DB3_HOST, DB3_NAME, etc. and restart the application")

def test_single_connection(db_name, db_config):
    """Test a single database connection"""
    async def test_connection():
        agent = DatabaseAgent(db_name, db_config)
        success = await agent.initialize()
        if success:
            tables = await agent.mcp_client.list_tables()
            await agent.cleanup()
            return success, len(tables)
        return False, 0
    
    with st.spinner(f"Testing connection to {db_name}..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success, table_count = loop.run_until_complete(test_connection())
        loop.close()
        
        if success:
            st.success(f"✅ Connection successful! Found {table_count} tables.")
        else:
            st.error(f"❌ Connection failed to {db_name}")

def test_connections():
    """Test all database connections"""
    config = st.session_state.config
    results = {}
    
    for db_name, db_config in config.DATABASES.items():
        with st.spinner(f"Testing {db_name}..."):
            async def test_connection():
                agent = DatabaseAgent(db_name, db_config)
                success = await agent.initialize()
                if success:
                    tables = await agent.mcp_client.list_tables()
                    await agent.cleanup()
                    return success, len(tables)
                return False, 0
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success, table_count = loop.run_until_complete(test_connection())
            loop.close()
            
            results[db_name] = {"success": success, "tables": table_count}
    
    # Display results
    st.subheader("🔍 Connection Test Results")
    for db_name, result in results.items():
        if result["success"]:
            st.success(f"✅ {db_name}: Connected successfully ({result['tables']} tables)")
        else:
            st.error(f"❌ {db_name}: Connection failed")

def show_analytics():
    """Show analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.query_history:
        st.info("No query history available. Execute some queries first!")
        return
    
    # Query statistics
    st.subheader("📈 Query Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_queries = len(st.session_state.query_history)
        st.metric("Total Queries", total_queries)
    
    with col2:
        successful_queries = sum(1 for q in st.session_state.query_history if q['result']['success'])
        st.metric("Successful Queries", successful_queries)
    
    with col3:
        failed_queries = total_queries - successful_queries
        st.metric("Failed Queries", failed_queries)
    
    with col4:
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Query timeline
    if len(st.session_state.query_history) > 1:
        st.subheader("📅 Query Timeline")
        
        df_history = pd.DataFrame([
            {
                'timestamp': q['timestamp'],
                'query': q['query'][:50] + "..." if len(q['query']) > 50 else q['query'],
                'success': q['result']['success'],
                'records': q['result'].get('data_summary', {}).get('total_records', 0) if q['result']['success'] else 0
            }
            for q in st.session_state.query_history
        ])
        
        # Success/failure over time
        fig = px.scatter(df_history, x='timestamp', y='success', 
                        color='success', size='records',
                        title="Query Success Over Time",
                        hover_data=['query'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Records retrieved over time
        successful_df = df_history[df_history['success'] == True]
        if not successful_df.empty:
            fig2 = px.line(successful_df, x='timestamp', y='records',
                          title="Records Retrieved Over Time",
                          markers=True)
            st.plotly_chart(fig2, use_container_width=True)

def show_history():
    """Show query history"""
    st.header("📋 Query History")
    
    if not st.session_state.query_history:
        st.info("No queries executed yet.")
        return
    
    # History controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search queries", placeholder="Enter search term...")
    
    with col2:
        show_successful = st.checkbox("✅ Successful only", value=False)
    
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
    
    # Filter history
    filtered_history = st.session_state.query_history.copy()
    
    if search_term:
        filtered_history = [
            q for q in filtered_history 
            if search_term.lower() in q['query'].lower()
        ]
    
    if show_successful:
        filtered_history = [
            q for q in filtered_history 
            if q['result']['success']
        ]
    
    # Display history
    st.subheader(f"📝 Query History ({len(filtered_history)} queries)")
    
    for i, query_info in enumerate(reversed(filtered_history)):
        idx = len(filtered_history) - i
        success_icon = "✅" if query_info['result']['success'] else "❌"
        
        with st.expander(f"{success_icon} Query #{idx}: {query_info['query'][:60]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Query:** {query_info['query']}")
                st.write(f"**Timestamp:** {query_info['timestamp']}")
            
            with col2:
                st.write(f"**Status:** {success_icon}")
                if query_info['result']['success'] and 'data_summary' in query_info['result']:
                    total_records = query_info['result']['data_summary'].get('total_records', 0)
                    st.write(f"**Records:** {total_records}")
            
            if query_info['result']['success']:
                st.write("**Insights:**")
                st.markdown(query_info['result']['insights'])
            else:
                st.write("**Error:**")
                st.error(query_info['result']['error'])
            
            # Re-execute button
            if st.button(f"🔄 Re-execute", key=f"rerun_{i}"):
                execute_query(query_info['query'])

if __name__ == "__main__":
    main()