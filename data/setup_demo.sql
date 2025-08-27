-- Demo SQLite database setup
-- This creates sample tables and data for testing the MCP server

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    merchant VARCHAR(100),
    description TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Insert sample users
INSERT INTO users (username, email, last_login, status) VALUES
('alice_johnson', 'alice@example.com', '2025-08-27 10:30:00', 'active'),
('bob_smith', 'bob@example.com', '2025-08-26 15:45:00', 'active'),
('carol_davis', 'carol@example.com', '2025-08-25 09:20:00', 'active'),
('dave_wilson', 'dave@example.com', '2025-08-24 14:10:00', 'inactive'),
('eve_brown', 'eve@example.com', '2025-08-27 11:00:00', 'active');

-- Insert sample transactions
INSERT INTO transactions (user_id, amount, category, merchant, description) VALUES
(1, 45.99, 'groceries', 'SuperMart', 'Weekly grocery shopping'),
(1, 12.50, 'coffee', 'Coffee Corner', 'Morning coffee'),
(1, 89.99, 'electronics', 'Tech Store', 'Wireless headphones'),
(2, 25.00, 'gas', 'Shell Station', 'Fuel for car'),
(2, 156.78, 'dining', 'Italian Bistro', 'Dinner with family'),
(2, 299.99, 'clothing', 'Fashion Hub', 'New jacket'),
(3, 67.45, 'groceries', 'Fresh Market', 'Organic vegetables'),
(3, 15.99, 'books', 'BookWorld', 'Programming book'),
(4, 199.99, 'electronics', 'Electronics Plus', 'Smartphone case'),
(5, 34.50, 'dining', 'Pizza Palace', 'Lunch meeting'),
(5, 78.90, 'health', 'Pharmacy Plus', 'Vitamins and supplements'),
(1, 500.00, 'utilities', 'City Power', 'Monthly electricity bill'),
(2, 1200.00, 'rent', 'Property Management', 'Monthly rent'),
(3, 89.99, 'entertainment', 'Cinema Complex', 'Movie tickets and snacks');