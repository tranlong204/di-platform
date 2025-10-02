#!/usr/bin/env python3
"""
Script to execute Supabase schema using REST API
"""

import requests
import json
import time

# Supabase configuration
SUPABASE_URL = "https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

def execute_sql(sql_statement, description):
    """Execute a SQL statement using Supabase REST API"""
    print(f"🔧 {description}...")
    
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use the Supabase REST API to execute SQL
    url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
    
    payload = {
        "sql": sql_statement
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"⚠️ {description} - Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def setup_supabase_schema():
    """Set up the complete Supabase schema"""
    print("🚀 Setting up Supabase database schema...")
    print(f"📊 Supabase URL: {SUPABASE_URL}")
    
    # SQL statements to execute
    sql_statements = [
        {
            "sql": "CREATE EXTENSION IF NOT EXISTS vector;",
            "description": "Enable pgvector extension"
        },
        {
            "sql": """
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
            """,
            "description": "Create documents table"
        },
        {
            "sql": """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(100) UNIQUE NOT NULL,
                document_id VARCHAR(50) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(1536),
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            "description": "Create document_chunks table"
        },
        {
            "sql": """
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_id VARCHAR(50) UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                response TEXT,
                processing_time FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """,
            "description": "Create queries table"
        },
        {
            "sql": """
            CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
            CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON document_chunks(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_queries_query_id ON queries(query_id);
            CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);
            """,
            "description": "Create indexes"
        },
        {
            "sql": """
            ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
            ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
            ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
            """,
            "description": "Enable Row Level Security"
        },
        {
            "sql": """
            CREATE POLICY "Service role can do everything" ON documents
                FOR ALL USING (auth.role() = 'service_role');
            CREATE POLICY "Service role can do everything" ON document_chunks
                FOR ALL USING (auth.role() = 'service_role');
            CREATE POLICY "Service role can do everything" ON queries
                FOR ALL USING (auth.role() = 'service_role');
            """,
            "description": "Create security policies"
        }
    ]
    
    success_count = 0
    total_count = len(sql_statements)
    
    for i, statement in enumerate(sql_statements, 1):
        print(f"\n[{i}/{total_count}] {statement['description']}")
        if execute_sql(statement['sql'], statement['description']):
            success_count += 1
        time.sleep(1)  # Small delay between statements
    
    print(f"\n📊 Schema Setup Summary:")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n🎉 Supabase schema setup completed successfully!")
        return True
    else:
        print("\n⚠️ Some operations failed. Please check the errors above.")
        return False

def test_connection():
    """Test the Supabase connection"""
    print("🔍 Testing Supabase connection...")
    
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    }
    
    # Test connection by querying the documents table
    url = f"{SUPABASE_URL}/rest/v1/documents?select=*&limit=1"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("✅ Supabase connection successful")
            data = response.json()
            print(f"📊 Documents table accessible: {len(data)} records found")
            return True
        else:
            print(f"⚠️ Connection test status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Supabase Schema Setup Script")
    print("=" * 50)
    
    # Test connection first
    if not test_connection():
        print("\n❌ Cannot connect to Supabase. Please check your credentials.")
        exit(1)
    
    # Setup schema
    success = setup_supabase_schema()
    
    if success:
        print("\n🎉 Database schema setup completed!")
        print("📝 Next steps:")
        print("1. Test document upload functionality")
        print("2. Resolve Vercel deployment protection")
        print("3. Verify chatbot functionality")
    else:
        print("\n❌ Schema setup failed!")
        print("🔧 Please check the errors above and try again.")
