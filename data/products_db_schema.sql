-- Products Database Schema
-- Contains product information, categories, pricing, and inventory

CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    parent_category_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_category_id) REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name VARCHAR(200) NOT NULL,
    description TEXT,
    category_id INTEGER NOT NULL,
    brand VARCHAR(100),
    model VARCHAR(100),
    sku VARCHAR(50) UNIQUE,
    price DECIMAL(10,2) NOT NULL,
    cost DECIMAL(10,2),
    weight DECIMAL(8,2),
    dimensions VARCHAR(50),
    color VARCHAR(50),
    size VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);

CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    warehouse_location VARCHAR(100),
    region VARCHAR(50),
    quantity_available INTEGER DEFAULT 0,
    quantity_reserved INTEGER DEFAULT 0,
    reorder_level INTEGER DEFAULT 10,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS product_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Sample Categories
INSERT INTO categories (category_name, description) VALUES
('Electronics', 'Electronic devices and gadgets'),
('Clothing', 'Apparel and fashion items'),
('Home & Garden', 'Home improvement and garden supplies'),
('Sports & Outdoors', 'Sports equipment and outdoor gear'),
('Books', 'Books and educational materials'),
('Health & Beauty', 'Health and beauty products');

-- Sample Products
INSERT INTO products (product_name, description, category_id, brand, model, sku, price, cost, weight, color, size, status) VALUES
('Wireless Headphones', 'High-quality noise-canceling headphones', 1, 'TechBrand', 'WH-1000XM5', 'TECH-WH-001', 299.99, 150.00, 0.25, 'Black', 'One Size', 'active'),
('Bluetooth Speaker', 'Portable waterproof speaker', 1, 'AudioPro', 'SP-200', 'TECH-SP-002', 89.99, 45.00, 0.50, 'Blue', 'Small', 'active'),
('Smartphone', 'Latest 5G smartphone', 1, 'PhoneCorp', 'P-2025', 'TECH-PH-003', 799.99, 400.00, 0.20, 'Silver', 'Standard', 'active'),
('Running Shoes', 'Professional running shoes', 2, 'SportFit', 'Runner-Pro', 'CLO-SH-001', 129.99, 65.00, 0.80, 'White', '10', 'active'),
('Casual T-Shirt', 'Cotton blend casual t-shirt', 2, 'FashionWear', 'Basic-Tee', 'CLO-TS-002', 24.99, 10.00, 0.20, 'Red', 'L', 'active'),
('Garden Tools Set', 'Complete gardening tool kit', 3, 'GardenPro', 'GT-Complete', 'HG-GT-001', 79.99, 35.00, 2.50, 'Green', 'Standard', 'active'),
('Indoor Plant Pot', 'Ceramic decorative plant pot', 3, 'HomeCraft', 'Pot-Classic', 'HG-PP-002', 19.99, 8.00, 1.20, 'Brown', 'Medium', 'active'),
('Tennis Racket', 'Professional tennis racket', 4, 'SportsPro', 'TR-Elite', 'SP-TR-001', 159.99, 80.00, 0.35, 'Yellow', 'Standard', 'active'),
('Programming Book', 'Learn Python programming', 5, 'TechBooks', 'Python-Guide', 'BK-PY-001', 49.99, 20.00, 0.60, 'Multi', 'Standard', 'active'),
('Vitamin Supplements', 'Daily vitamin complex', 6, 'HealthPlus', 'Daily-Vita', 'HB-VT-001', 29.99, 12.00, 0.15, 'White', 'Bottle', 'active');

-- Sample Inventory
INSERT INTO inventory (product_id, warehouse_location, region, quantity_available, quantity_reserved, reorder_level) VALUES
(1, 'Seattle Warehouse', 'West Coast', 150, 10, 25),
(1, 'Chicago Warehouse', 'Midwest', 120, 5, 25),
(1, 'Miami Warehouse', 'Southeast', 80, 8, 20),
(2, 'Seattle Warehouse', 'West Coast', 200, 15, 30),
(2, 'Denver Warehouse', 'Mountain', 90, 5, 20),
(3, 'Boston Warehouse', 'Northeast', 75, 20, 15),
(3, 'Phoenix Warehouse', 'Southwest', 110, 12, 20),
(4, 'Chicago Warehouse', 'Midwest', 180, 25, 40),
(4, 'Miami Warehouse', 'Southeast', 160, 18, 35),
(5, 'Seattle Warehouse', 'West Coast', 300, 30, 50),
(6, 'Denver Warehouse', 'Mountain', 95, 8, 20),
(7, 'Boston Warehouse', 'Northeast', 140, 15, 30),
(8, 'Phoenix Warehouse', 'Southwest', 85, 10, 20),
(9, 'Chicago Warehouse', 'Midwest', 220, 25, 40),
(10, 'Miami Warehouse', 'Southeast', 190, 20, 35);

-- Sample Product Reviews
INSERT INTO product_reviews (product_id, user_id, rating, review_text) VALUES
(1, 1, 5, 'Amazing sound quality and comfort!'),
(1, 3, 4, 'Great headphones, worth the price'),
(2, 2, 5, 'Perfect for outdoor activities'),
(3, 5, 4, 'Fast performance, good camera'),
(4, 4, 5, 'Very comfortable for long runs'),
(5, 6, 3, 'Good quality, fits well'),
(6, 7, 5, 'Complete set, very useful'),
(7, 1, 4, 'Beautiful pot, good quality'),
(8, 2, 5, 'Professional grade racket'),
(9, 3, 5, 'Excellent book for beginners');