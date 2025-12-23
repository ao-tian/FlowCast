-- Airport Operations Database Schema

-- Passenger Traffic Table
CREATE TABLE IF NOT EXISTS passenger_traffic (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    passenger_count INTEGER NOT NULL CHECK (passenger_count >= 0),
    airport_code VARCHAR(10),
    terminal VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, airport_code, terminal)
);

CREATE INDEX idx_passenger_traffic_date ON passenger_traffic(date);
CREATE INDEX idx_passenger_traffic_airport ON passenger_traffic(airport_code);

-- Weather Data Table
CREATE TABLE IF NOT EXISTS weather_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    temperature_celsius DECIMAL(5,2),
    precipitation_mm DECIMAL(5,2),
    humidity_percent DECIMAL(5,2),
    weather_condition VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

CREATE INDEX idx_weather_date ON weather_data(date);

-- Holidays/Events Table
CREATE TABLE IF NOT EXISTS holidays (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    holiday_name VARCHAR(200),
    holiday_type VARCHAR(50),
    is_national BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, holiday_name)
);

CREATE INDEX idx_holidays_date ON holidays(date);

-- Retail Sales Table (Synthetic)
CREATE TABLE IF NOT EXISTS retail_sales (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    sales_amount DECIMAL(12,2) NOT NULL CHECK (sales_amount >= 0),
    sales_count INTEGER NOT NULL CHECK (sales_count >= 0),
    category VARCHAR(100),
    terminal VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_retail_sales_date ON retail_sales(date);
CREATE INDEX idx_retail_sales_category ON retail_sales(category);

-- Forecasts Table
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    forecast_value DECIMAL(10,2) NOT NULL,
    lower_bound DECIMAL(10,2),
    upper_bound DECIMAL(10,2),
    confidence_interval DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_forecasts_date ON forecasts(date);
CREATE INDEX idx_forecasts_model ON forecasts(model_name);

-- Model Performance Table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,4),
    evaluation_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_performance_model ON model_performance(model_name);
CREATE INDEX idx_model_performance_date ON model_performance(evaluation_date);

-- Data Quality Checks Table
CREATE TABLE IF NOT EXISTS data_quality_checks (
    id SERIAL PRIMARY KEY,
    check_name VARCHAR(200) NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    message TEXT,
    rows_checked INTEGER,
    rows_failed INTEGER,
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_quality_date ON data_quality_checks(check_date);
CREATE INDEX idx_data_quality_status ON data_quality_checks(status);

