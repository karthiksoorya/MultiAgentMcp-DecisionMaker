-- Enhanced Multi-Agent PostgreSQL Analysis System - Sample Database
-- Initialize test database with comprehensive sample data for development and testing

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with enhanced schema
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    user_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    date_of_birth DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    login_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'deleted')),
    user_type VARCHAR(20) DEFAULT 'regular' CHECK (user_type IN ('regular', 'premium', 'admin')),
    country_code CHAR(2),
    city VARCHAR(100),
    signup_source VARCHAR(50),
    referral_code VARCHAR(20),
    email_verified BOOLEAN DEFAULT FALSE,
    phone_verified BOOLEAN DEFAULT FALSE
);

-- Create indexes for better performance
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_last_login ON users(last_login);
CREATE INDEX idx_users_user_type ON users(user_type);

-- Transactions table with enhanced schema
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    transaction_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10, 2) NOT NULL CHECK (amount >= 0),
    currency CHAR(3) DEFAULT 'USD',
    category VARCHAR(50),
    subcategory VARCHAR(50),
    merchant VARCHAR(100),
    merchant_category VARCHAR(50),
    description TEXT,
    payment_method VARCHAR(20) CHECK (payment_method IN ('credit_card', 'debit_card', 'paypal', 'bank_transfer', 'crypto')),
    transaction_type VARCHAR(20) DEFAULT 'purchase' CHECK (transaction_type IN ('purchase', 'refund', 'subscription', 'transfer')),
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    is_recurring BOOLEAN DEFAULT FALSE,
    location_city VARCHAR(100),
    location_country CHAR(2),
    device_type VARCHAR(20),
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for transactions
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_transactions_category ON transactions(category);
CREATE INDEX idx_transactions_merchant ON transactions(merchant);
CREATE INDEX idx_transactions_status ON transactions(status);

-- User sessions table for behavior tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id SERIAL PRIMARY KEY,
    session_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_end TIMESTAMP,
    duration_seconds INTEGER,
    pages_viewed INTEGER DEFAULT 0,
    actions_taken INTEGER DEFAULT 0,
    device_type VARCHAR(20),
    browser VARCHAR(50),
    os VARCHAR(50),
    ip_address INET,
    referrer_url TEXT,
    exit_page VARCHAR(200)
);

CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_start ON user_sessions(session_start);

-- Product catalog for transaction context
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_uuid UUID DEFAULT uuid_generate_v4() UNIQUE,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50),
    subcategory VARCHAR(50),
    price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Transaction items for detailed purchase tracking
CREATE TABLE IF NOT EXISTS transaction_items (
    item_id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER DEFAULT 1,
    unit_price DECIMAL(10, 2),
    total_price DECIMAL(10, 2)
);

-- Insert comprehensive sample data

-- Insert sample users with diverse profiles
INSERT INTO users (
    username, email, first_name, last_name, date_of_birth, last_login, login_count, 
    status, user_type, country_code, city, signup_source, email_verified, phone_verified
) VALUES
-- Active premium users
('john_doe', 'john.doe@email.com', 'John', 'Doe', '1985-03-15', NOW() - INTERVAL '2 hours', 150, 'active', 'premium', 'US', 'New York', 'organic', true, true),
('jane_smith', 'jane.smith@email.com', 'Jane', 'Smith', '1992-07-22', NOW() - INTERVAL '1 day', 89, 'active', 'premium', 'US', 'San Francisco', 'referral', true, true),
('alice_johnson', 'alice.j@email.com', 'Alice', 'Johnson', '1988-11-10', NOW() - INTERVAL '3 days', 234, 'active', 'regular', 'CA', 'Toronto', 'social_media', true, false),

-- Regular active users  
('bob_wilson', 'bob.wilson@email.com', 'Bob', 'Wilson', '1995-01-08', NOW() - INTERVAL '1 week', 45, 'active', 'regular', 'US', 'Chicago', 'google_ads', true, true),
('charlie_davis', 'charlie.d@email.com', 'Charlie', 'Davis', '1990-09-03', NOW() - INTERVAL '4 days', 78, 'active', 'regular', 'UK', 'London', 'organic', false, false),
('diana_prince', 'diana.p@email.com', 'Diana', 'Prince', '1987-06-18', NOW() - INTERVAL '2 days', 112, 'active', 'regular', 'US', 'Los Angeles', 'referral', true, true),

-- Less active users
('eve_brown', 'eve.brown@email.com', 'Eve', 'Brown', '1993-12-25', NOW() - INTERVAL '2 weeks', 23, 'active', 'regular', 'AU', 'Sydney', 'social_media', true, false),
('frank_miller', 'frank.m@email.com', 'Frank', 'Miller', '1982-04-12', NOW() - INTERVAL '1 month', 67, 'inactive', 'regular', 'US', 'Miami', 'organic', true, true),

-- Inactive/churned users
('grace_lee', 'grace.lee@email.com', 'Grace', 'Lee', '1991-08-30', NOW() - INTERVAL '3 months', 12, 'inactive', 'regular', 'KR', 'Seoul', 'google_ads', false, false),
('henry_garcia', 'henry.g@email.com', 'Henry', 'Garcia', '1989-02-14', NOW() - INTERVAL '6 months', 5, 'inactive', 'regular', 'MX', 'Mexico City', 'referral', true, false),

-- New users
('iris_wang', 'iris.wang@email.com', 'Iris', 'Wang', '1996-10-05', NOW() - INTERVAL '3 hours', 3, 'active', 'regular', 'CN', 'Shanghai', 'social_media', false, false),
('jack_taylor', 'jack.t@email.com', 'Jack', 'Taylor', '1994-05-20', NOW() - INTERVAL '1 day', 8, 'active', 'regular', 'US', 'Seattle', 'organic', true, false);

-- Insert sample products
INSERT INTO products (name, category, subcategory, price) VALUES
('Wireless Headphones', 'electronics', 'audio', 199.99),
('Coffee Subscription', 'food', 'beverages', 29.99),
('Fitness Tracker', 'electronics', 'wearables', 149.99),
('Premium Pizza', 'food', 'restaurant', 24.99),
('Cloud Storage Plan', 'software', 'subscription', 9.99),
('Running Shoes', 'clothing', 'footwear', 129.99),
('Smartphone Case', 'electronics', 'accessories', 19.99),
('Organic Groceries', 'food', 'groceries', 85.50),
('Online Course', 'education', 'subscription', 99.99),
('Movie Tickets', 'entertainment', 'cinema', 15.00);

-- Insert comprehensive transaction data with realistic patterns
INSERT INTO transactions (
    user_id, transaction_date, amount, currency, category, subcategory, merchant, merchant_category,
    description, payment_method, transaction_type, status, location_city, location_country, device_type
) VALUES
-- John Doe (Premium user, high activity) - Recent transactions
(1, NOW() - INTERVAL '2 hours', 199.99, 'USD', 'electronics', 'audio', 'TechStore Pro', 'electronics', 'Wireless Headphones', 'credit_card', 'purchase', 'completed', 'New York', 'US', 'desktop'),
(1, NOW() - INTERVAL '1 day', 24.99, 'USD', 'food', 'restaurant', 'Pizza Palace', 'restaurant', 'Premium Pizza Delivery', 'credit_card', 'purchase', 'completed', 'New York', 'US', 'mobile'),
(1, NOW() - INTERVAL '3 days', 149.99, 'USD', 'electronics', 'wearables', 'FitnessTech', 'electronics', 'Fitness Tracker', 'credit_card', 'purchase', 'completed', 'New York', 'US', 'mobile'),
(1, NOW() - INTERVAL '1 week', 29.99, 'USD', 'food', 'subscription', 'Coffee Co.', 'subscription', 'Monthly Coffee Subscription', 'credit_card', 'subscription', 'completed', 'New York', 'US', 'desktop'),

-- Jane Smith (Premium user, regular activity)  
(2, NOW() - INTERVAL '1 day', 129.99, 'USD', 'clothing', 'footwear', 'SportShop', 'retail', 'Running Shoes', 'paypal', 'purchase', 'completed', 'San Francisco', 'US', 'mobile'),
(2, NOW() - INTERVAL '4 days', 85.50, 'USD', 'food', 'groceries', 'Organic Market', 'grocery', 'Weekly Groceries', 'debit_card', 'purchase', 'completed', 'San Francisco', 'US', 'mobile'),
(2, NOW() - INTERVAL '1 week', 9.99, 'USD', 'software', 'subscription', 'CloudStore', 'software', 'Monthly Cloud Storage', 'credit_card', 'subscription', 'completed', 'San Francisco', 'US', 'desktop'),

-- Alice Johnson (Regular user, very active)
(3, NOW() - INTERVAL '2 days', 19.99, 'CAD', 'electronics', 'accessories', 'Phone Accessories Plus', 'electronics', 'Smartphone Case', 'credit_card', 'purchase', 'completed', 'Toronto', 'CA', 'mobile'),
(3, NOW() - INTERVAL '5 days', 99.99, 'CAD', 'education', 'online_course', 'EduPlatform', 'education', 'Data Science Course', 'paypal', 'purchase', 'completed', 'Toronto', 'CA', 'desktop'),
(3, NOW() - INTERVAL '1 week', 15.00, 'CAD', 'entertainment', 'cinema', 'MovieMax', 'entertainment', '2x Movie Tickets', 'credit_card', 'purchase', 'completed', 'Toronto', 'CA', 'mobile'),

-- Bob Wilson (Regular user, moderate activity)
(4, NOW() - INTERVAL '1 week', 45.50, 'USD', 'food', 'restaurant', 'Burger Junction', 'restaurant', 'Family Dinner', 'credit_card', 'purchase', 'completed', 'Chicago', 'US', 'mobile'),
(4, NOW() - INTERVAL '2 weeks', 12.99, 'USD', 'transport', 'rideshare', 'RideNow', 'transportation', 'Airport Trip', 'paypal', 'purchase', 'completed', 'Chicago', 'US', 'mobile'),

-- Charlie Davis (Regular user, moderate activity)
(5, NOW() - INTERVAL '4 days', 67.80, 'GBP', 'food', 'groceries', 'Local Supermarket', 'grocery', 'Weekly Shopping', 'debit_card', 'purchase', 'completed', 'London', 'UK', 'mobile'),
(5, NOW() - INTERVAL '1 week', 25.00, 'GBP', 'entertainment', 'streaming', 'StreamService', 'entertainment', 'Monthly Subscription', 'credit_card', 'subscription', 'completed', 'London', 'UK', 'desktop'),

-- Diana Prince (Regular user, active)
(6, NOW() - INTERVAL '3 days', 89.99, 'USD', 'clothing', 'apparel', 'Fashion Forward', 'retail', 'Designer T-Shirt', 'credit_card', 'purchase', 'completed', 'Los Angeles', 'US', 'mobile'),
(6, NOW() - INTERVAL '1 week', 55.25, 'USD', 'health', 'wellness', 'Wellness Center', 'health', 'Yoga Class Package', 'credit_card', 'purchase', 'completed', 'Los Angeles', 'US', 'mobile'),

-- Historical transactions for analysis
(1, NOW() - INTERVAL '1 month', 299.99, 'USD', 'electronics', 'computing', 'TechStore Pro', 'electronics', 'Laptop Accessories', 'credit_card', 'purchase', 'completed', 'New York', 'US', 'desktop'),
(1, NOW() - INTERVAL '2 months', 49.99, 'USD', 'software', 'subscription', 'ProductivityApp', 'software', 'Annual Subscription', 'credit_card', 'subscription', 'completed', 'New York', 'US', 'desktop'),
(2, NOW() - INTERVAL '1 month', 175.00, 'USD', 'health', 'fitness', 'GymChain', 'health', 'Personal Training Session', 'credit_card', 'purchase', 'completed', 'San Francisco', 'US', 'mobile'),
(3, NOW() - INTERVAL '1 month', 125.50, 'CAD', 'travel', 'accommodation', 'HotelBooking', 'travel', 'Weekend Getaway', 'credit_card', 'purchase', 'completed', 'Toronto', 'CA', 'desktop');

-- Insert sample user sessions
INSERT INTO user_sessions (
    user_id, session_start, session_end, duration_seconds, pages_viewed, actions_taken, 
    device_type, browser, os, referrer_url
) VALUES
(1, NOW() - INTERVAL '2 hours', NOW() - INTERVAL '1 hour 45 minutes', 900, 12, 5, 'desktop', 'Chrome', 'Windows', 'https://google.com'),
(1, NOW() - INTERVAL '1 day', NOW() - INTERVAL '23 hours 30 minutes', 1800, 8, 3, 'mobile', 'Safari', 'iOS', 'https://facebook.com'),
(2, NOW() - INTERVAL '1 day', NOW() - INTERVAL '23 hours 45 minutes', 600, 5, 2, 'mobile', 'Chrome', 'Android', 'https://instagram.com'),
(3, NOW() - INTERVAL '3 days', NOW() - INTERVAL '2 days 23 hours', 2400, 15, 8, 'desktop', 'Firefox', 'macOS', 'https://twitter.com'),
(4, NOW() - INTERVAL '1 week', NOW() - INTERVAL '6 days 23 hours', 1200, 7, 4, 'mobile', 'Chrome', 'Android', 'https://google.com');

-- Create views for common analytics queries
CREATE OR REPLACE VIEW user_transaction_summary AS
SELECT 
    u.user_id,
    u.username,
    u.email,
    u.status,
    u.user_type,
    u.created_at as user_created_at,
    u.last_login,
    COUNT(t.transaction_id) as total_transactions,
    COALESCE(SUM(t.amount), 0) as total_spent,
    COALESCE(AVG(t.amount), 0) as avg_transaction_amount,
    MAX(t.transaction_date) as last_transaction_date,
    EXTRACT(EPOCH FROM (NOW() - MAX(t.transaction_date)))/86400 as days_since_last_transaction
FROM users u
LEFT JOIN transactions t ON u.user_id = t.user_id AND t.status = 'completed'
GROUP BY u.user_id, u.username, u.email, u.status, u.user_type, u.created_at, u.last_login;

-- Create view for category spending analysis
CREATE OR REPLACE VIEW category_spending_analysis AS
SELECT 
    category,
    subcategory,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MIN(amount) as min_amount,
    MAX(amount) as max_amount,
    COUNT(DISTINCT user_id) as unique_users
FROM transactions 
WHERE status = 'completed'
GROUP BY category, subcategory
ORDER BY total_amount DESC;

-- Create view for user behavior metrics
CREATE OR REPLACE VIEW user_behavior_metrics AS
SELECT 
    u.user_id,
    u.username,
    u.status,
    u.user_type,
    EXTRACT(EPOCH FROM (NOW() - u.created_at))/86400 as account_age_days,
    CASE 
        WHEN u.last_login > NOW() - INTERVAL '7 days' THEN 'highly_active'
        WHEN u.last_login > NOW() - INTERVAL '30 days' THEN 'active'  
        WHEN u.last_login > NOW() - INTERVAL '90 days' THEN 'at_risk'
        ELSE 'churned'
    END as activity_segment,
    u.login_count,
    COALESCE(s.avg_session_duration, 0) as avg_session_duration,
    COALESCE(s.total_sessions, 0) as total_sessions,
    COALESCE(t.total_transactions, 0) as total_transactions,
    COALESCE(t.total_spent, 0) as total_spent
FROM users u
LEFT JOIN (
    SELECT user_id, 
           AVG(duration_seconds) as avg_session_duration,
           COUNT(*) as total_sessions
    FROM user_sessions 
    GROUP BY user_id
) s ON u.user_id = s.user_id
LEFT JOIN (
    SELECT user_id,
           COUNT(*) as total_transactions,
           SUM(amount) as total_spent
    FROM transactions 
    WHERE status = 'completed'
    GROUP BY user_id
) t ON u.user_id = t.user_id;

-- Add some utility functions
CREATE OR REPLACE FUNCTION get_user_segment(user_id_param INTEGER)
RETURNS TEXT AS $$
DECLARE
    segment TEXT;
BEGIN
    SELECT activity_segment INTO segment 
    FROM user_behavior_metrics 
    WHERE user_id = user_id_param;
    
    RETURN COALESCE(segment, 'unknown');
END;
$$ LANGUAGE plpgsql;

-- Performance optimization: Update statistics
ANALYZE users;
ANALYZE transactions;
ANALYZE user_sessions;
ANALYZE products;