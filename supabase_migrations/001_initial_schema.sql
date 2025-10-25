-- Supabase Migration: Initial Schema for Document Intelligence Platform
-- This migration creates all the necessary tables for the DI Platform

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    title VARCHAR(255),
    summary TEXT,
    metadata_json JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on content_hash for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_processed ON documents(processed);

-- Document chunks table
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    token_count INTEGER NOT NULL,
    embedding_id VARCHAR(100),
    metadata_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for document_chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_id ON document_chunks(embedding_id);

-- Queries table
CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_text TEXT,
    query_hash VARCHAR(64) NOT NULL,
    context_documents JSONB,
    context_chunks JSONB,
    similarity_scores JSONB,
    processing_time FLOAT,
    user_feedback INTEGER CHECK (user_feedback >= 1 AND user_feedback <= 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for queries
CREATE INDEX IF NOT EXISTS idx_queries_query_hash ON queries(query_hash);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);

-- Agents table
CREATE TABLE IF NOT EXISTS agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    workflow_config JSONB NOT NULL,
    tools JSONB,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for agents
CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(active);
CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name);

-- Agent executions table
CREATE TABLE IF NOT EXISTS agent_executions (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    query_id INTEGER NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    execution_steps JSONB,
    final_result TEXT,
    execution_time FLOAT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for agent_executions
CREATE INDEX IF NOT EXISTS idx_agent_executions_agent_id ON agent_executions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_query_id ON agent_executions(query_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_success ON agent_executions(success);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agents_updated_at 
    BEFORE UPDATE ON agents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS) for better security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_executions ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users (you can customize these based on your needs)
CREATE POLICY "Allow all operations for authenticated users" ON documents
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON document_chunks
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON queries
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON agents
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow all operations for authenticated users" ON agent_executions
    FOR ALL USING (auth.role() = 'authenticated');

-- Create a function to get document statistics
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE (
    total_documents BIGINT,
    processed_documents BIGINT,
    total_chunks BIGINT,
    total_queries BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM documents) as total_documents,
        (SELECT COUNT(*) FROM documents WHERE processed = TRUE) as processed_documents,
        (SELECT COUNT(*) FROM document_chunks) as total_chunks,
        (SELECT COUNT(*) FROM queries) as total_queries;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
