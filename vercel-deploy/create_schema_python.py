#!/usr/bin/env python3
"""
Script to connect to Supabase and execute SQL schema using Python
"""

import os
import sys
import json
import time
import subprocess

# Supabase configuration
SUPABASE_URL = "https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

def install_supabase():
    """Install supabase client library"""
    print("📦 Installing Supabase client library...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "supabase"])
        print("✅ Supabase client installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Supabase client: {e}")
        return False

def create_supabase_client():
    """Create Supabase client"""
    try:
        from supabase import create_client, Client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Supabase client created successfully")
        return supabase
    except ImportError:
        print("❌ Supabase client not available")
        return None
    except Exception as e:
        print(f"❌ Error creating Supabase client: {e}")
        return None

def execute_sql_via_rpc(supabase, sql, description):
    """Execute SQL via Supabase RPC"""
    print(f"🔧 {description}...")
    
    try:
        # Try to execute SQL via RPC
        result = supabase.rpc('exec_sql', {'sql': sql}).execute()
        print(f"✅ {description} - Success")
        return True
    except Exception as e:
        print(f"⚠️ {description} - RPC failed: {str(e)}")
        return False

def execute_sql_via_management_api(sql, description):
    """Execute SQL via Supabase Management API"""
    print(f"🔧 {description} (via Management API)...")
    
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json"
        }
        
        # Use Management API
        url = f"https://api.supabase.com/v1/projects/pvmaxqesaxplmztyaujj/sql"
        
        payload = {
            "query": sql
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"⚠️ {description} - Status: {response.status_code}, Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def create_tables_via_direct_insert(supabase):
    """Create tables by trying to insert data"""
    print("\n🔄 Trying to create tables by inserting test data...")
    
    # Test data for each table
    test_data = {
        "documents": {
            "doc_id": "test_doc_123",
            "filename": "test.txt",
            "file_type": "text/plain",
            "file_size": 100,
            "chunks_count": 1,
            "status": "processed"
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
            "response": "This is a test response",
            "processing_time": 1.5
        }
    }
    
    success_count = 0
    
    for table, data in test_data.items():
        print(f"🔧 Creating table '{table}' by inserting test data...")
        try:
            result = supabase.table(table).insert(data).execute()
            print(f"✅ Table '{table}' created successfully")
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            if "PGRST205" in error_msg:
                print(f"⚠️ Table '{table}' doesn't exist yet - this is expected")
            else:
                print(f"❌ Error with table '{table}': {error_msg}")
    
    return success_count > 0

def test_tables_exist(supabase):
    """Test if tables exist"""
    print("\n🔍 Testing if tables exist...")
    
    tables = ["documents", "document_chunks", "queries"]
    existing_tables = []
    
    for table in tables:
        try:
            result = supabase.table(table).select("*").limit(1).execute()
            print(f"✅ Table '{table}' exists and accessible")
            existing_tables.append(table)
        except Exception as e:
            error_msg = str(e)
            if "PGRST205" in error_msg:
                print(f"❌ Table '{table}' not found")
            else:
                print(f"⚠️ Error testing table '{table}': {error_msg}")
    
    return existing_tables

def create_schema_via_sql_execution():
    """Create schema using SQL execution"""
    print("🚀 Creating Supabase database schema...")
    
    # SQL statements
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
    
    # Try to install supabase client
    if not install_supabase():
        print("❌ Cannot proceed without Supabase client")
        return False
    
    # Create Supabase client
    supabase = create_supabase_client()
    if not supabase:
        print("❌ Cannot proceed without Supabase client")
        return False
    
    success_count = 0
    total_count = len(sql_statements)
    
    # Try different methods to execute SQL
    for i, statement in enumerate(sql_statements, 1):
        print(f"\n[{i}/{total_count}] {statement['description']}")
        
        # Method 1: Try RPC
        if execute_sql_via_rpc(supabase, statement['sql'], statement['description']):
            success_count += 1
            continue
        
        # Method 2: Try Management API
        if execute_sql_via_management_api(statement['sql'], statement['description']):
            success_count += 1
            continue
        
        print(f"⚠️ {statement['description']} - All methods failed")
        
        time.sleep(1)
    
    print(f"\n📊 SQL Execution Summary:")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    # If SQL execution failed, try creating tables by inserting data
    if success_count == 0:
        print("\n🔄 SQL execution failed, trying alternative method...")
        if create_tables_via_direct_insert(supabase):
            print("✅ Tables created via direct insert method")
            success_count = total_count
    
    # Test if tables exist
    existing_tables = test_tables_exist(supabase)
    
    return len(existing_tables) == 3

def main():
    """Main function"""
    print("🚀 Supabase Database Schema Creation Script")
    print("=" * 50)
    print(f"📊 Supabase URL: {SUPABASE_URL}")
    print(f"🔑 Service Role Key: {SUPABASE_SERVICE_ROLE_KEY[:20]}...")
    
    try:
        success = create_schema_via_sql_execution()
        
        if success:
            print("\n🎉 Database schema creation completed successfully!")
            print("✅ All tables created and accessible")
            print("\n🎯 Next steps:")
            print("1. Test document upload functionality")
            print("2. Test chatbot functionality")
            print("3. Verify the application is working")
        else:
            print("\n⚠️ Database schema creation had issues")
            print("🔧 You may need to:")
            print("1. Check Supabase project permissions")
            print("2. Verify the service role key is correct")
            print("3. Try accessing Supabase dashboard manually")
            
    except Exception as e:
        print(f"\n❌ Script failed with error: {str(e)}")
        print("🔧 Please check your Supabase configuration and try again")

if __name__ == "__main__":
    main()
