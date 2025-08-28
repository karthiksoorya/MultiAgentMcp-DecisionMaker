-- Sales Database Schema
-- Contains orders, transactions, payments, and sales analytics

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    order_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    order_status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(12,2) NOT NULL,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    shipping_amount DECIMAL(8,2) DEFAULT 0,
    discount_amount DECIMAL(8,2) DEFAULT 0,
    payment_method VARCHAR(50),
    shipping_address TEXT,
    billing_address TEXT,
    region VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(12,2) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    user_id INTEGER NOT NULL,
    transaction_type VARCHAR(20) DEFAULT 'purchase',
    amount DECIMAL(12,2) NOT NULL,
    payment_method VARCHAR(50),
    payment_status VARCHAR(20) DEFAULT 'completed',
    transaction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    region VARCHAR(50),
    merchant VARCHAR(100),
    category VARCHAR(50),
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS sales_analytics (
    analytics_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    region VARCHAR(50),
    product_category VARCHAR(100),
    total_sales DECIMAL(15,2),
    total_orders INTEGER,
    unique_customers INTEGER,
    avg_order_value DECIMAL(10,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Sample Orders
INSERT INTO orders (user_id, order_date, order_status, total_amount, tax_amount, shipping_amount, payment_method, region) VALUES
(1, '2025-08-25 10:30:00', 'completed', 329.98, 26.40, 9.99, 'credit_card', 'West Coast'),
(2, '2025-08-25 14:15:00', 'completed', 154.98, 12.40, 5.99, 'paypal', 'Midwest'),
(3, '2025-08-26 09:20:00', 'completed', 99.98, 8.00, 7.99, 'credit_card', 'Southeast'),
(4, '2025-08-26 16:45:00', 'completed', 189.97, 15.20, 8.99, 'debit_card', 'Mountain'),
(5, '2025-08-27 08:10:00', 'processing', 849.97, 67.98, 12.99, 'credit_card', 'Northeast'),
(6, '2025-08-27 11:30:00', 'completed', 79.98, 6.40, 5.99, 'paypal', 'Southwest'),
(7, '2025-08-27 15:20:00', 'shipped', 49.98, 4.00, 6.99, 'credit_card', 'West Coast'),
(1, '2025-08-27 18:45:00', 'completed', 89.99, 7.20, 5.99, 'credit_card', 'West Coast'),
(3, '2025-08-27 20:15:00', 'pending', 179.98, 14.40, 8.99, 'paypal', 'Southeast');

-- Sample Order Items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 299.99, 299.99),
(1, 7, 1, 19.99, 19.99),
(2, 4, 1, 129.99, 129.99),
(2, 5, 1, 24.99, 24.99),
(3, 6, 1, 79.99, 79.99),
(3, 7, 1, 19.99, 19.99),
(4, 8, 1, 159.99, 159.99),
(4, 10, 1, 29.99, 29.99),
(5, 3, 1, 799.99, 799.99),
(5, 9, 1, 49.99, 49.99),
(6, 6, 1, 79.99, 79.99),
(7, 9, 1, 49.99, 49.99),
(8, 2, 1, 89.99, 89.99),
(9, 4, 1, 129.99, 129.99),
(9, 10, 1, 29.99, 29.99),
(9, 7, 1, 19.99, 19.99);

-- Sample Transactions
INSERT INTO transactions (order_id, user_id, transaction_type, amount, payment_method, payment_status, transaction_date, region, merchant, category, description) VALUES
(1, 1, 'purchase', 329.98, 'credit_card', 'completed', '2025-08-25 10:30:00', 'West Coast', 'TechStore', 'electronics', 'Headphones and plant pot'),
(2, 2, 'purchase', 154.98, 'paypal', 'completed', '2025-08-25 14:15:00', 'Midwest', 'SportWear', 'clothing', 'Running shoes and t-shirt'),
(3, 3, 'purchase', 99.98, 'credit_card', 'completed', '2025-08-26 09:20:00', 'Southeast', 'HomeDepot', 'home_garden', 'Garden tools and pot'),
(4, 4, 'purchase', 189.97, 'debit_card', 'completed', '2025-08-26 16:45:00', 'Mountain', 'SportZone', 'sports', 'Tennis racket and vitamins'),
(5, 5, 'purchase', 849.97, 'credit_card', 'processing', '2025-08-27 08:10:00', 'Northeast', 'TechWorld', 'electronics', 'Smartphone and book'),
(6, 6, 'purchase', 79.98, 'paypal', 'completed', '2025-08-27 11:30:00', 'Southwest', 'HomeDepot', 'home_garden', 'Garden tools'),
(7, 7, 'purchase', 49.98, 'credit_card', 'completed', '2025-08-27 15:20:00', 'West Coast', 'BookWorld', 'books', 'Programming book'),
(8, 1, 'purchase', 89.99, 'credit_card', 'completed', '2025-08-27 18:45:00', 'West Coast', 'AudioStore', 'electronics', 'Bluetooth speaker'),
(9, 3, 'purchase', 179.98, 'paypal', 'pending', '2025-08-27 20:15:00', 'Southeast', 'SportWear', 'clothing', 'Shoes, vitamins, pot');

-- Sample Sales Analytics
INSERT INTO sales_analytics (date, region, product_category, total_sales, total_orders, unique_customers, avg_order_value) VALUES
('2025-08-25', 'West Coast', 'electronics', 329.98, 1, 1, 329.98),
('2025-08-25', 'Midwest', 'clothing', 154.98, 1, 1, 154.98),
('2025-08-26', 'Southeast', 'home_garden', 99.98, 1, 1, 99.98),
('2025-08-26', 'Mountain', 'sports', 189.97, 1, 1, 189.97),
('2025-08-27', 'Northeast', 'electronics', 849.97, 1, 1, 849.97),
('2025-08-27', 'Southwest', 'home_garden', 79.98, 1, 1, 79.98),
('2025-08-27', 'West Coast', 'books', 49.98, 1, 1, 49.98),
('2025-08-27', 'West Coast', 'electronics', 89.99, 1, 1, 89.99),
('2025-08-27', 'Southeast', 'clothing', 179.98, 1, 1, 179.98);