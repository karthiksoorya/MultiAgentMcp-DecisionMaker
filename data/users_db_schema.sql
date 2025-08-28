-- Users Database Schema
-- Contains user information, demographics, and registration data

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    date_of_birth DATE,
    phone VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    status VARCHAR(20) DEFAULT 'active',
    user_type VARCHAR(20) DEFAULT 'regular'
);

CREATE TABLE IF NOT EXISTS user_addresses (
    address_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    address_type VARCHAR(20) DEFAULT 'primary',
    street_address VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    region VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    category VARCHAR(50),
    preference_value VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Sample Users Data
INSERT INTO users (username, email, first_name, last_name, date_of_birth, phone, last_login, status, user_type) VALUES
('alice_johnson', 'alice@example.com', 'Alice', 'Johnson', '1992-03-15', '+1-555-0101', '2025-08-27 10:30:00', 'active', 'premium'),
('bob_smith', 'bob@example.com', 'Bob', 'Smith', '1988-07-22', '+1-555-0102', '2025-08-26 15:45:00', 'active', 'regular'),
('carol_davis', 'carol@example.com', 'Carol', 'Davis', '1995-11-08', '+1-555-0103', '2025-08-25 09:20:00', 'active', 'premium'),
('dave_wilson', 'dave@example.com', 'Dave', 'Wilson', '1990-01-12', '+1-555-0104', '2025-08-24 14:10:00', 'inactive', 'regular'),
('eve_brown', 'eve@example.com', 'Eve', 'Brown', '1993-09-25', '+1-555-0105', '2025-08-27 11:00:00', 'active', 'premium'),
('frank_miller', 'frank@example.com', 'Frank', 'Miller', '1987-05-18', '+1-555-0106', '2025-08-26 16:30:00', 'active', 'regular'),
('grace_lee', 'grace@example.com', 'Grace', 'Lee', '1994-12-03', '+1-555-0107', '2025-08-27 08:45:00', 'active', 'premium');

-- User Addresses
INSERT INTO user_addresses (user_id, address_type, street_address, city, state, postal_code, country, region) VALUES
(1, 'primary', '123 Maple St', 'Seattle', 'WA', '98101', 'USA', 'West Coast'),
(2, 'primary', '456 Oak Ave', 'Chicago', 'IL', '60601', 'USA', 'Midwest'),
(3, 'primary', '789 Pine Rd', 'Miami', 'FL', '33101', 'USA', 'Southeast'),
(4, 'primary', '321 Elm St', 'Denver', 'CO', '80201', 'USA', 'Mountain'),
(5, 'primary', '654 Cedar Ln', 'Boston', 'MA', '02101', 'USA', 'Northeast'),
(6, 'primary', '987 Birch Dr', 'Phoenix', 'AZ', '85001', 'USA', 'Southwest'),
(7, 'primary', '147 Spruce Way', 'Portland', 'OR', '97201', 'USA', 'West Coast');

-- User Preferences
INSERT INTO user_preferences (user_id, category, preference_value) VALUES
(1, 'product_category', 'electronics'),
(1, 'price_range', 'premium'),
(2, 'product_category', 'clothing'),
(2, 'price_range', 'budget'),
(3, 'product_category', 'home_garden'),
(3, 'price_range', 'mid_range'),
(4, 'product_category', 'sports'),
(4, 'price_range', 'budget'),
(5, 'product_category', 'electronics'),
(5, 'price_range', 'premium'),
(6, 'product_category', 'books'),
(6, 'price_range', 'budget'),
(7, 'product_category', 'health_beauty'),
(7, 'price_range', 'premium');