#!/usr/bin/env python3
"""
Script to create Supabase database tables using Management API
"""

import subprocess
import json
import time

# Supabase configuration
SUPABASE_URL = "https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

def execute_sql_via_curl(sql_command, description):
    """Execute SQL command via curl"""
    print(f"🔧 {description}...")
    
    headers = [
        "-H", f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", "Content-Type: application/json"
    ]
    
    # Use the Supabase REST API to execute SQL via RPC
    url = f"{SUPABASE_URL}/rest/v1/rpc/exec"
    
    payload = {
        "sql": sql_command
    }
    
    try:
        cmd = [
            "curl", "-s", "-X", "POST", url,
            *headers,
            "-d", json.dumps(payload)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            if response and not response.startswith('{"code":'):
                print(f"✅ {description} - Success")
                return True
            else:
                print(f"⚠️ {description} - Response: {response}")
                return False
        else:
            print(f"❌ {description} - Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - Timeout")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def create_tables_via_api():
    """Create tables using Supabase REST API"""
    print("🚀 Creating Supabase tables via API...")
    
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
        if execute_sql_via_curl(statement['sql'], statement['description']):
            success_count += 1
        time.sleep(2)  # Delay between statements
    
    print(f"\n📊 Schema Creation Summary:")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    return success_count == total_count

def test_tables_created():
    """Test if tables were created successfully"""
    print("\n🔍 Testing if tables were created...")
    
    headers = [
        "-H", f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    ]
    
    # Test each table
    tables = ["documents", "document_chunks", "queries"]
    
    for table in tables:
        url = f"{SUPABASE_URL}/rest/v1/{table}?select=*&limit=1"
        
        try:
            cmd = ["curl", "-s", *headers, url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if '"code":"PGRST205"' in response:
                    print(f"❌ Table '{table}' not found")
                else:
                    print(f"✅ Table '{table}' exists and accessible")
            else:
                print(f"❌ Error testing table '{table}': {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error testing table '{table}': {str(e)}")

def create_tables_via_direct_insert():
    """Alternative method: Create tables by trying to insert test data"""
    print("\n🔄 Trying alternative method: Direct table creation...")
    
    headers = [
        "-H", f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", "Content-Type: application/json"
    ]
    
    # Try to create tables by inserting test data
    test_data = {
        "documents": {
            "doc_id": "test_doc_123",
            "filename": "test.txt",
            "file_type": "text/plain",
            "file_size": 100,
            "chunks_count": 1
        },
        "document_chunks": {
            "chunk_id": "test_chunk_123",
            "document_id": "test_doc_123",
            "chunk_index": 0,
            "content": "This is a test chunk"
        },
        "queries": {
            "query_id": "test_query_123",
            "query_text": "What is this about?",
            "response": "This is a test response"
        }
    }
    
    for table, data in test_data.items():
        url = f"{SUPABASE_URL}/rest/v1/{table}"
        
        try:
            cmd = [
                "curl", "-s", "-X", "POST", url,
                *headers,
                "-d", json.dumps(data)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                if '"code":"PGRST205"' in response:
                    print(f"⚠️ Table '{table}' doesn't exist yet")
                else:
                    print(f"✅ Successfully inserted test data into '{table}'")
            else:
                print(f"❌ Error with table '{table}': {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error with table '{table}': {str(e)}")

if __name__ == "__main__":
    print("🚀 Supabase Database Table Creation Script")
    print("=" * 50)
    
    # Method 1: Try to create tables via SQL execution
    print("📋 Method 1: Creating tables via SQL execution...")
    success = create_tables_via_api()
    
    if not success:
        print("\n📋 Method 2: Trying direct table creation...")
        create_tables_via_direct_insert()
    
    # Test if tables were created
    test_tables_created()
    
    print("\n🎯 Next Steps:")
    print("1. Check if tables were created successfully")
    print("2. Test document upload functionality")
    print("3. Test chatbot functionality")
    
    if success:
        print("\n🎉 Database schema creation completed!")
    else:
        print("\n⚠️ Some operations may have failed. Please check the output above.")
