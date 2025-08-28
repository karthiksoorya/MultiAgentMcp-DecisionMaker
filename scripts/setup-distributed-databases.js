#!/usr/bin/env node

/**
 * Setup script for distributed databases
 * Creates separate SQLite databases for Users, Products, and Sales
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🗄️ Setting up Distributed Database Architecture...');
console.log('='.repeat(60));

// Ensure directories exist
const dataDir = './data';
const scriptsDir = './scripts';

if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
    console.log('✅ Created data directory');
}

if (!fs.existsSync(scriptsDir)) {
    fs.mkdirSync(scriptsDir, { recursive: true });
    console.log('✅ Created scripts directory');
}

// Database configurations
const databases = [
    {
        name: 'Users Database',
        file: 'users_db.sqlite',
        schema: 'users_db_schema.sql',
        description: 'Contains user information, demographics, and addresses'
    },
    {
        name: 'Products Database', 
        file: 'products_db.sqlite',
        schema: 'products_db_schema.sql',
        description: 'Contains product catalog, categories, and inventory'
    },
    {
        name: 'Sales Database',
        file: 'sales_db.sqlite', 
        schema: 'sales_db_schema.sql',
        description: 'Contains orders, transactions, and sales analytics'
    }
];

console.log('\n📋 Database Setup Plan:');
databases.forEach((db, index) => {
    console.log(`${index + 1}. ${db.name} (${db.file})`);
    console.log(`   ${db.description}`);
});

console.log('\n🔧 Creating databases...');

// Check if sqlite3 is available
let sqliteCommand = 'sqlite3';
try {
    execSync('sqlite3 -version', { stdio: 'pipe' });
    console.log('✅ SQLite3 found');
} catch (error) {
    console.log('⚠️  SQLite3 not found in PATH - databases will be created automatically by MCP servers');
    sqliteCommand = null;
}

// Create each database
databases.forEach((db, index) => {
    const dbPath = path.join(dataDir, db.file);
    const schemaPath = path.join(dataDir, db.schema);
    
    console.log(`\n${index + 1}. Setting up ${db.name}...`);
    
    // Check if schema file exists
    if (!fs.existsSync(schemaPath)) {
        console.log(`   ❌ Schema file not found: ${schemaPath}`);
        return;
    }
    
    // Create database if SQLite is available
    if (sqliteCommand) {
        try {
            // Remove existing database
            if (fs.existsSync(dbPath)) {
                fs.unlinkSync(dbPath);
                console.log(`   🗑️  Removed existing ${db.file}`);
            }
            
            // Create new database with schema
            const command = `${sqliteCommand} "${dbPath}" < "${schemaPath}"`;
            execSync(command, { stdio: 'pipe' });
            console.log(`   ✅ Created ${db.file} with sample data`);
            
            // Verify database was created
            const stats = fs.statSync(dbPath);
            console.log(`   📊 Database size: ${(stats.size / 1024).toFixed(2)} KB`);
            
        } catch (error) {
            console.log(`   ❌ Failed to create ${db.file}: ${error.message}`);
        }
    } else {
        console.log(`   ⏭️  ${db.file} will be created by MCP server on first use`);
    }
});

console.log('\n🚀 MCP Server Configuration:');
console.log('Users MCP Server will use port: Default MCP protocol');
console.log('Products MCP Server will use port: Default MCP protocol');  
console.log('Sales MCP Server will use port: Default MCP protocol');

console.log('\n📋 Next Steps:');
console.log('1. Run: npm run start-distributed');
console.log('2. This will start:');
console.log('   - Users MCP Server (./data/users_db.sqlite)');
console.log('   - Products MCP Server (./data/products_db.sqlite)');
console.log('   - Sales MCP Server (./data/sales_db.sqlite)');
console.log('   - Streamlit Web Interface (http://localhost:8501)');

console.log('\n✅ Distributed database setup complete!');
console.log('='.repeat(60));