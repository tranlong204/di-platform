-- Database initialization script
-- This script sets up the PostgreSQL database with pgvector extension

-- Create the database if it doesn't exist
-- (This is handled by the POSTGRES_DB environment variable)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables (these will be created by SQLAlchemy, but we can add any custom setup here)

-- Create indexes for better performance
-- (These will be created after tables are created by the application)

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE di_platform TO di_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO di_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO di_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO di_user;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO di_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO di_user;
