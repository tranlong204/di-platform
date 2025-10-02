#!/usr/bin/env python3
"""
Simple script to set up Supabase schema using direct SQL execution
"""

import subprocess
import json
import os

def run_sql_via_psql(sql_command, description):
    """Run SQL command via psql if available"""
    print(f"🔧 {description}...")
    
    # Supabase connection string
    conn_string = "postgresql://postgres:[YOUR-PASSWORD]@db.pvmaxqesaxplmztyaujj.supabase.co:5432/postgres"
    
    try:
        # Try to run via psql
        result = subprocess.run([
            "psql", conn_string, "-c", sql_command
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - Timeout")
        return False
    except FileNotFoundError:
        print(f"⚠️ {description} - psql not found, trying alternative method")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def create_schema_file():
    """Create a schema file that can be executed manually"""
    schema_sql = """
-- Supabase Database Schema for Document Intelligence Platform
-- Execute this in your Supabase SQL Editor

-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(50) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_size INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'processed',
    chunks_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create document_chunks table with vector support
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(100) UNIQUE NOT NULL,
    document_id VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI text-embedding-3-small dimension
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create queries table
CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(50) UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    response TEXT,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON document_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_queries_query_id ON queries(query_id);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);

-- Enable Row Level Security (RLS) for better security
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;

-- Create policies to allow service role access
CREATE POLICY "Service role can do everything" ON documents
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything" ON document_chunks
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything" ON queries
    FOR ALL USING (auth.role() = 'service_role');
"""
    
    with open("supabase_schema_final.sql", "w") as f:
        f.write(schema_sql)
    
    print("📄 Created supabase_schema_final.sql file")
    return True

def test_supabase_connection():
    """Test connection to Supabase using curl"""
    print("🔍 Testing Supabase connection...")
    
    url = "https://pvmaxqesaxplmztyaujj.supabase.co/rest/v1/"
    headers = {
        "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"
    }
    
    try:
        result = subprocess.run([
            "curl", "-s", "-H", f"apikey: {headers['apikey']}", 
            "-H", f"Authorization: {headers['Authorization']}", 
            url
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Supabase connection successful")
            return True
        else:
            print(f"❌ Connection failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Supabase Schema Setup")
    print("=" * 40)
    
    # Test connection
    if not test_supabase_connection():
        print("\n❌ Cannot connect to Supabase")
        exit(1)
    
    # Create schema file
    create_schema_file()
    
    print("\n📋 Manual Setup Required:")
    print("1. Go to your Supabase dashboard: https://supabase.com/dashboard/project/pvmaxqesaxplmztyaujj")
    print("2. Navigate to SQL Editor")
    print("3. Copy and paste the contents of 'supabase_schema_final.sql'")
    print("4. Click 'Run' to execute the schema")
    print("\n🎯 This will create all necessary tables and indexes for your application!")
