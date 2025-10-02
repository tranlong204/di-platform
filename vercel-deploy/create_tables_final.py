#!/usr/bin/env python3
"""
Final attempt to create Supabase tables using REST API with custom functions
"""

import subprocess
import json
import time

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

def create_custom_function():
    """Create a custom function to execute SQL"""
    print("🔧 Creating custom SQL execution function...")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type: application/json"
    ]
    
    # Try to create a function that can execute SQL
    function_sql = """
    CREATE OR REPLACE FUNCTION create_di_platform_tables()
    RETURNS text AS $$
    BEGIN
        -- Enable pgvector extension
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
        
        -- Create document_chunks table
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
        
        -- Create queries table
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(50) UNIQUE NOT NULL,
            query_text TEXT NOT NULL,
            response TEXT,
            processing_time FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        -- Enable Row Level Security
        ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
        ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE queries ENABLE ROW LEVEL SECURITY;
        
        -- Create policies
        CREATE POLICY "Service role can do everything" ON documents
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON document_chunks
            FOR ALL USING (auth.role() = 'service_role');
        CREATE POLICY "Service role can do everything" ON queries
            FOR ALL USING (auth.role() = 'service_role');
        
        RETURN 'Tables created successfully';
    END;
    $$ LANGUAGE plpgsql;
    """
    
    # Try to execute this via RPC
    url = f"{SUPABASE_URL}/rest/v1/rpc/exec"
    payload = {"sql": function_sql}
    data = json.dumps(payload)
    
    success, response, error = execute_curl_command(url, headers, data, "POST")
    
    if success:
        print(f"Response: {response}")
        if "PGRST202" not in response:
            print("✅ Custom function created successfully")
            return True
        else:
            print("⚠️ Custom function creation failed")
    else:
        print(f"❌ Error creating custom function: {error}")
    
    return False

def execute_custom_function():
    """Execute the custom function"""
    print("🔧 Executing custom function to create tables...")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type: application/json"
    ]
    
    url = f"{SUPABASE_URL}/rest/v1/rpc/create_di_platform_tables"
    payload = {}
    data = json.dumps(payload)
    
    success, response, error = execute_curl_command(url, headers, data, "POST")
    
    if success:
        print(f"Response: {response}")
        if "Tables created successfully" in response:
            print("✅ Tables created successfully via custom function")
            return True
        else:
            print("⚠️ Custom function execution failed")
    else:
        print(f"❌ Error executing custom function: {error}")
    
    return False

def test_tables_final():
    """Final test of tables"""
    print("\n🔍 Final test of tables...")
    
    headers = [
        f"apikey: {SUPABASE_SERVICE_ROLE_KEY}",
        f"Authorization: Bearer {SUPABASE_SERVICE_ROLE_KEY}"
    ]
    
    tables = ["documents", "document_chunks", "queries"]
    existing_tables = []
    
    for table in tables:
        url = f"{SUPABASE_URL}/rest/v1/{table}?select=*&limit=1"
        
        success, response, error = execute_curl_command(url, headers)
        
        if success:
            if "PGRST205" in response:
                print(f"❌ Table '{table}' not found")
            else:
                print(f"✅ Table '{table}' exists and accessible")
                existing_tables.append(table)
        else:
            print(f"❌ Error testing table '{table}': {error}")
    
    return existing_tables

def main():
    """Main function"""
    print("🚀 Final Supabase Table Creation Attempt")
    print("=" * 50)
    
    # Try to create custom function
    if create_custom_function():
        # Try to execute the function
        if execute_custom_function():
            print("🎉 Tables created successfully!")
        else:
            print("⚠️ Function execution failed")
    else:
        print("⚠️ Custom function creation failed")
    
    # Test tables
    existing_tables = test_tables_final()
    
    print(f"\n📊 Final Results:")
    print(f"✅ Tables created: {len(existing_tables)}/3")
    print(f"📋 Existing tables: {existing_tables}")
    
    if len(existing_tables) == 3:
        print("\n🎉 SUCCESS! All tables created!")
        print("✅ Your application should now work properly")
        print("\n🔗 Test your application at:")
        print("https://vercel-deploy-qxic2yjqn-long-trans-projects-9092b735.vercel.app")
    else:
        print("\n❌ FAILED: Tables were not created")
        print("\n🔧 Alternative Solutions:")
        print("1. Use Supabase CLI: supabase db reset")
        print("2. Access Supabase dashboard when it's working")
        print("3. Contact Supabase support")
        print("4. Use a different database provider")
        
        print("\n📄 Manual SQL Schema:")
        print("Copy and paste this into Supabase SQL Editor when accessible:")
        print("-" * 50)
        print("""
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
        """)

if __name__ == "__main__":
    main()
