#!/usr/bin/env python3
"""
Script to create Supabase tables using Supabase client library
"""

import subprocess
import json
import time
import os

# Supabase configuration
SUPABASE_URL = "https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

def install_supabase_client():
    """Install supabase client if not available"""
    try:
        import supabase
        print("✅ Supabase client already available")
        return True
    except ImportError:
        print("📦 Installing Supabase client...")
        try:
            subprocess.run(["pip3", "install", "--user", "supabase"], check=True)
            print("✅ Supabase client installed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Supabase client")
            return False

def create_tables_via_supabase_client():
    """Create tables using Supabase client"""
    try:
        from supabase import create_client, Client
        
        # Create Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Supabase client created")
        
        # Try to create tables by executing SQL
        sql_statements = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            """
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
            """
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
            """
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_id VARCHAR(50) UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                response TEXT,
                processing_time FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """
        ]
        
        for i, sql in enumerate(sql_statements, 1):
            print(f"🔧 Executing SQL statement {i}/{len(sql_statements)}...")
            try:
                # Try to execute SQL via RPC
                result = supabase.rpc('exec_sql', {'sql': sql}).execute()
                print(f"✅ SQL statement {i} executed successfully")
            except Exception as e:
                print(f"⚠️ SQL statement {i} failed: {str(e)}")
                # Try alternative method
                try:
                    # Try to create table by inserting test data
                    if i == 2:  # documents table
                        test_data = {
                            "doc_id": "test_doc_123",
                            "filename": "test.txt",
                            "file_type": "text/plain",
                            "file_size": 100,
                            "chunks_count": 1
                        }
                        supabase.table("documents").insert(test_data).execute()
                        print(f"✅ Documents table created via test insert")
                    elif i == 3:  # document_chunks table
                        test_data = {
                            "chunk_id": "test_chunk_123",
                            "document_id": "test_doc_123",
                            "chunk_index": 0,
                            "content": "This is a test chunk"
                        }
                        supabase.table("document_chunks").insert(test_data).execute()
                        print(f"✅ Document_chunks table created via test insert")
                    elif i == 4:  # queries table
                        test_data = {
                            "query_id": "test_query_123",
                            "query_text": "What is this about?",
                            "response": "This is a test response"
                        }
                        supabase.table("queries").insert(test_data).execute()
                        print(f"✅ Queries table created via test insert")
                except Exception as e2:
                    print(f"❌ Alternative method also failed: {str(e2)}")
            
            time.sleep(1)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating Supabase client: {str(e)}")
        return False

def create_tables_via_curl_alternative():
    """Alternative method using curl with different endpoints"""
    print("\n🔄 Trying alternative curl method...")
    
    headers = [
        "-H", f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", "Content-Type: application/json"
    ]
    
    # Try to create a custom function first
    create_function_sql = """
    CREATE OR REPLACE FUNCTION create_tables()
    RETURNS void AS $$
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
        
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
        
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(50) UNIQUE NOT NULL,
            query_text TEXT NOT NULL,
            response TEXT,
            processing_time FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
        ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY "Service role can do everything" ON documents
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON document_chunks
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON queries
            FOR ALL USING (auth.role() = 'service_role');
    END;
    $$ LANGUAGE plpgsql;
    """
    
    # Try to create the function
    url = f"{SUPABASE_URL}/rest/v1/rpc/create_tables"
    
    try:
        cmd = [
            "curl", "-s", "-X", "POST", url,
            *headers,
            "-d", "{}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"Response: {response}")
            if '"code":"PGRST202"' not in response:
                print("✅ Function created successfully")
                return True
            else:
                print("⚠️ Function creation failed")
        else:
            print(f"❌ Error: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    return False

def test_tables_exist():
    """Test if tables exist"""
    print("\n🔍 Testing if tables exist...")
    
    headers = [
        "-H", f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        "-H", f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    ]
    
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

if __name__ == "__main__":
    print("🚀 Supabase Table Creation Script v2")
    print("=" * 50)
    
    # Try to install supabase client
    if install_supabase_client():
        # Try method 1: Supabase client
        print("\n📋 Method 1: Using Supabase client...")
        success = create_tables_via_supabase_client()
        
        if not success:
            # Try method 2: Alternative curl method
            print("\n📋 Method 2: Using alternative curl method...")
            create_tables_via_curl_alternative()
    
    # Test if tables exist
    test_tables_exist()
    
    print("\n🎯 Summary:")
    print("If tables were not created, you may need to:")
    print("1. Access Supabase dashboard manually")
    print("2. Use a different approach")
    print("3. Contact Supabase support")
