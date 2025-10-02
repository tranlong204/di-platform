#!/usr/bin/env python3
"""
Simple script to create Supabase tables using only standard library and curl
"""

import subprocess
import json
import time
import os

# Supabase configuration
SUPABASE_URL = "https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

def execute_curl_command(url, headers, data=None, method="GET"):
    """Execute curl command and return response"""
    cmd = ["curl", "-s", "-X", method]
    
    for header in headers:
        cmd.extend(["-H", header])
    
    if data:
        cmd.extend(["-d", data])
    
    cmd.append(url)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout.strip(), result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def test_supabase_connection():
    """Test connection to Supabase"""
    print("🔍 Testing Supabase connection...")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    ]
    
    url = f"{SUPABASE_URL}/rest/v1/"
    
    success, response, error = execute_curl_command(url, headers)
    
    if success:
        print("✅ Supabase connection successful")
        return True
    else:
        print(f"❌ Supabase connection failed: {error}")
        return False

def create_table_via_insert(table_name, test_data, description):
    """Create table by trying to insert test data"""
    print(f"🔧 {description}...")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type: application/json"
    ]
    
    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    data = json.dumps(test_data)
    
    success, response, error = execute_curl_command(url, headers, data, "POST")
    
    if success:
        if "PGRST205" in response:
            print(f"⚠️ Table '{table_name}' doesn't exist yet - this is expected")
            return False
        elif "error" in response.lower() or "Error" in response:
            print(f"❌ Error creating table '{table_name}': {response}")
            return False
        else:
            print(f"✅ Successfully inserted test data into '{table_name}'")
            return True
    else:
        print(f"❌ Failed to create table '{table_name}': {error}")
        return False

def test_table_exists(table_name):
    """Test if table exists"""
    print(f"🔍 Testing table: {table_name}")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    ]
    
    url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*&limit=1"
    
    success, response, error = execute_curl_command(url, headers)
    
    if success:
        if "PGRST205" in response:
            print(f"❌ Table '{table_name}' not found")
            return False
        else:
            print(f"✅ Table '{table_name}' exists and accessible")
            return True
    else:
        print(f"❌ Error testing table '{table_name}': {error}")
        return False

def create_tables_via_management_api():
    """Create tables using Supabase Management API"""
    print("🚀 Creating tables via Supabase Management API...")
    
    headers = [
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type: application/json"
    ]
    
    # SQL statements to execute
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
        """,
        """
        ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
        ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
        """,
        """
        CREATE POLICY "Service role can do everything" ON documents
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON document_chunks
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON queries
            FOR ALL USING (auth.role() = 'service_role');
        """
    ]
    
    url = "https://api.supabase.com/v1/projects/pvmaxqesaxplmztyaujj/sql"
    
    success_count = 0
    
    for i, sql in enumerate(sql_statements, 1):
        print(f"\n[{i}/{len(sql_statements)}] Executing SQL statement {i}...")
        
        payload = {"query": sql}
        data = json.dumps(payload)
        
        success, response, error = execute_curl_command(url, headers, data, "POST")
        
        if success:
            if "message" in response and "invalid signature" in response:
                print(f"⚠️ SQL statement {i} - Invalid signature (permission issue)")
            elif "message" in response and "Bad control character" in response:
                print(f"⚠️ SQL statement {i} - JSON parsing error")
            else:
                print(f"✅ SQL statement {i} - Success")
                success_count += 1
        else:
            print(f"❌ SQL statement {i} - Failed: {error}")
        
        time.sleep(1)
    
    print(f"\n📊 Management API Summary:")
    print(f"✅ Successful: {success_count}/{len(sql_statements)}")
    print(f"❌ Failed: {len(sql_statements) - success_count}/{len(sql_statements)}")
    
    return success_count > 0

def create_tables_via_direct_insert():
    """Create tables by inserting test data"""
    print("\n🔄 Creating tables via direct insert method...")
    
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
        if create_table_via_insert(table, data, f"Creating table '{table}'"):
            success_count += 1
    
    print(f"\n📊 Direct Insert Summary:")
    print(f"✅ Successful: {success_count}/{len(test_data)}")
    print(f"❌ Failed: {len(test_data) - success_count}/{len(test_data)}")
    
    return success_count > 0

def main():
    """Main function"""
    print("🚀 Supabase Database Schema Creation Script (Standard Library)")
    print("=" * 60)
    print(f"📊 Supabase URL: {SUPABASE_URL}")
    print(f"🔑 Service Role Key: {SUPABASE_SERVICE_ROLE_KEY[:20]}...")
    
    # Test connection first
    if not test_supabase_connection():
        print("\n❌ Cannot connect to Supabase. Please check your configuration.")
        return
    
    # Method 1: Try Management API
    print("\n📋 Method 1: Using Supabase Management API...")
    api_success = create_tables_via_management_api()
    
    # Method 2: Try direct insert
    if not api_success:
        print("\n📋 Method 2: Using direct insert method...")
        insert_success = create_tables_via_direct_insert()
    else:
        insert_success = True
    
    # Test if tables exist
    print("\n🔍 Testing if tables exist...")
    tables = ["documents", "document_chunks", "queries"]
    existing_tables = []
    
    for table in tables:
        if test_table_exists(table):
            existing_tables.append(table)
    
    # Summary
    print(f"\n📊 Final Summary:")
    print(f"✅ Tables created: {len(existing_tables)}/{len(tables)}")
    print(f"📋 Existing tables: {existing_tables}")
    
    if len(existing_tables) == len(tables):
        print("\n🎉 All tables created successfully!")
        print("✅ Your application should now work properly")
        print("\n🎯 Next steps:")
        print("1. Test document upload functionality")
        print("2. Test chatbot functionality")
        print("3. Verify the application is working")
    else:
        print("\n⚠️ Some tables were not created")
        print("🔧 You may need to:")
        print("1. Check Supabase project permissions")
        print("2. Verify the service role key is correct")
        print("3. Try accessing Supabase dashboard manually")
        print("4. Contact Supabase support")

if __name__ == "__main__":
    main()
