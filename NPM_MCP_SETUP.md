# 🔌 NPM Package MCP Server Setup

## Overview

This project now uses the **official npm package** from [executeautomation/mcp-database-server](https://github.com/executeautomation/mcp-database-server) for standardized MCP database operations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Your Application                        │
├─────────────────────────────────────────────────────────┤
│  🌐 Streamlit Web Interface (Python)                   │
│  📊 Multi-Agent Analytics & Orchestration              │
│  🗄️ Database Management & Connection Pooling           │
└─────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────┐
│            NPM Package Integration                      │
├─────────────────────────────────────────────────────────┤
│  📦 @executeautomation/database-server                 │
│  🔌 Official MCP Protocol Implementation               │
│  🛠️ Standardized Database Tools                        │
└─────────────────────────────────────────────────────────┘
                              ↓
                         Claude Integration
```

## 🚀 Installation & Setup

### 1. Install Dependencies
```bash
# Python dependencies
pip install -r requirements.txt

# NPM dependencies (includes MCP server)
npm install
```

### 2. Available NPM Scripts
```bash
# SQLite Demo (no database setup required)
npm run mcp-sqlite

# PostgreSQL with environment variables
npm run mcp-postgres-env

# Start both web interface and MCP server
npm run start-both-sqlite    # SQLite demo
npm run start-both           # PostgreSQL with .env
```

## 🎯 Launch Options

### Option 1: Demo with SQLite (Recommended for Testing)
```bash
# Windows
start_demo.bat

# Contains:
# - Automatic SQLite database creation
# - Sample data (users, transactions)
# - MCP server + Streamlit web interface
```

### Option 2: Production with PostgreSQL
```bash
# Setup your .env file first
cp .env.template .env
# Edit .env with your database credentials

# Windows
start_npm_mcp.bat

# Linux/Mac  
./start_npm_mcp.sh
```

### Option 3: Direct NPM Scripts
```bash
# Just MCP server (SQLite)
npm run mcp-sqlite

# Just MCP server (PostgreSQL) 
npm run mcp-postgres-env

# Both services together
npm run start-both-sqlite
```

## 🗄️ MCP Database Tools Available

The npm package provides these standardized tools for Claude:

| Tool | Purpose | Example |
|------|---------|---------|
| `read_query` | Execute SELECT statements | `SELECT * FROM users WHERE status = 'active'` |
| `write_query` | INSERT/UPDATE/DELETE | `INSERT INTO users (username, email) VALUES ('john', 'john@example.com')` |
| `create_table` | Create new tables | `CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT)` |
| `list_tables` | Show all tables | Returns list of all tables in database |
| `describe_table` | Table schema info | Shows columns, types, constraints for a table |
| `export_query` | Export to CSV/JSON | Export query results in specified format |
| `append_insight` | Add business insight | Store analysis insights for later reference |
| `list_insights` | Show all insights | Retrieve all stored business insights |

## 📁 File Structure

```
├── package.json              # NPM configuration with MCP server
├── node_modules/             # NPM dependencies including MCP server
├── data/
│   ├── demo.db              # SQLite demo database (auto-created)
│   └── setup_demo.sql       # Sample data setup
├── start_demo.bat           # Windows demo launcher
├── start_npm_mcp.bat        # Windows production launcher  
├── start_npm_mcp.sh         # Linux/Mac production launcher
└── streamlit_app.py         # Python web interface
```

## 🎮 Usage Examples

### Claude Interaction Examples:
```
Claude: "Can you list all tables in the database?"
→ Uses list_tables tool

Claude: "Show me all active users"
→ Uses read_query with "SELECT * FROM users WHERE status = 'active'"

Claude: "Export the user data to CSV"
→ Uses export_query with format="csv"

Claude: "Add insight: Users are most active on weekends"
→ Uses append_insight tool
```

## ✅ Benefits of NPM Package Approach

1. **🏢 Official & Maintained**: Actively maintained by executeautomation team
2. **📋 Standardized**: Follows official MCP protocol specifications
3. **🔒 Secure**: Built-in validation and security features
4. **🚀 Production Ready**: Tested in production environments
5. **🔧 Easy Updates**: Update with `npm update @executeautomation/database-server`
6. **🤝 Compatible**: Works with any MCP-compatible AI system
7. **📚 Documented**: Official documentation and examples

## 🔍 Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check if PostgreSQL is running
# Verify .env file has correct credentials
# Test connection manually first
```

### SQLite Demo Not Working
```bash
# Recreate demo database
npm run create-demo-db
rm data/demo.db  # Remove existing
npm run mcp-sqlite  # Will create new one
```

### MCP Server Not Starting
```bash
# Check if port is available
# Verify npm dependencies installed
npm install
```

## 🎯 Next Steps

1. **Test Demo**: Run `start_demo.bat` to try SQLite demo
2. **Setup Production**: Configure `.env` for PostgreSQL
3. **Customize**: Modify npm scripts for your specific needs
4. **Integrate**: Connect Claude to your MCP server for AI database operations

The system now provides a **production-ready, standardized MCP database server** using the official npm package! 🚀