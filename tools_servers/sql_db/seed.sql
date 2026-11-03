-- Synthetic data for the SentinelMCP sandbox. All people, emails and amounts are FAKE.
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    city        TEXT NOT NULL,
    plan        TEXT NOT NULL CHECK (plan IN ('free', 'pro', 'team')),
    signup_date TEXT NOT NULL
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    product     TEXT NOT NULL,
    amount_inr  INTEGER NOT NULL CHECK (amount_inr >= 0),
    status      TEXT NOT NULL CHECK (status IN ('paid', 'refunded', 'pending')),
    order_date  TEXT NOT NULL
);

CREATE TABLE tmp_import (
    row_id  INTEGER PRIMARY KEY,
    payload TEXT NOT NULL
);

INSERT INTO customers VALUES
 (1001, 'Avery Testperson',  'avery.t@customers.example',  'Pune',      'pro',  '2025-01-14'),
 (1002, 'Jordan Sampleton',  'jordan.s@customers.example', 'Mumbai',    'free', '2025-02-03'),
 (1003, 'Riley Fakename',    'riley.f@customers.example',  'Bengaluru', 'team', '2025-02-21'),
 (1004, 'Casey Placeholder', 'casey.p@customers.example',  'Delhi',     'pro',  '2025-03-08'),
 (1005, 'Morgan Dummy',      'morgan.d@customers.example', 'Hyderabad', 'free', '2025-03-30'),
 (1006, 'Taylor Mockwell',   'taylor.m@customers.example', 'Chennai',   'team', '2025-04-12'),
 (1007, 'Quinn Exampleby',   'quinn.e@customers.example',  'Kolkata',   'pro',  '2025-05-01'),
 (1008, 'Skyler Synthetic',  'skyler.s@customers.example', 'Pune',      'free', '2025-05-19'),
 (1009, 'Drew Notreal',      'drew.n@customers.example',   'Jaipur',    'pro',  '2025-06-07'),
 (1010, 'Rowan Specimen',    'rowan.s@customers.example',  'Ahmedabad', 'team', '2025-06-25');

INSERT INTO orders VALUES
 (1, 1001, 'pro-annual',     9990, 'paid',     '2025-01-14'),
 (2, 1003, 'team-monthly',   2490, 'paid',     '2025-02-21'),
 (3, 1003, 'team-monthly',   2490, 'paid',     '2025-03-21'),
 (4, 1004, 'pro-monthly',     999, 'paid',     '2025-03-08'),
 (5, 1004, 'pro-monthly',     999, 'refunded', '2025-04-08'),
 (6, 1006, 'team-annual',   24900, 'paid',     '2025-04-12'),
 (7, 1007, 'pro-monthly',     999, 'paid',     '2025-05-01'),
 (8, 1007, 'pro-monthly',     999, 'paid',     '2025-06-01'),
 (9, 1009, 'pro-annual',     9990, 'pending',  '2025-06-07'),
 (10, 1010, 'team-monthly',  2490, 'paid',     '2025-06-25'),
 (11, 1001, 'export-addon',   499, 'paid',     '2025-07-02'),
 (12, 1010, 'team-monthly',  2490, 'paid',     '2025-07-25');

INSERT INTO tmp_import VALUES
 (1, 'stale row from last import'),
 (2, 'stale row from last import');
