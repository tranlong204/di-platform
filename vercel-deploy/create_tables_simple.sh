#!/bin/bash
# Simple script to create Supabase tables by inserting test data

echo "🚀 Creating Supabase tables by inserting test data..."

# Supabase configuration
SUPABASE_URL="https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

# Headers
HEADERS=(-H "apikey: $SUPABASE_SERVICE_ROLE_KEY" -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" -H "Content-Type: application/json")

# Function to test table existence
test_table() {
    local table="$1"
    echo "🔍 Testing table: $table"
    
    response=$(curl -s "${HEADERS[@]}" "$SUPABASE_URL/rest/v1/$table?select=*&limit=1")
    
    if [[ $response == *"PGRST205"* ]]; then
        echo "❌ Table '$table' not found"
        return 1
    else
        echo "✅ Table '$table' exists and accessible"
        return 0
    fi
}

# Function to create table by inserting test data
create_table_by_insert() {
    local table="$1"
    local data="$2"
    local description="$3"
    
    echo "🔧 $description..."
    
    response=$(curl -s -X POST "${HEADERS[@]}" "$SUPABASE_URL/rest/v1/$table" -d "$data")
    
    if [[ $response == *"PGRST205"* ]]; then
        echo "⚠️ Table '$table' doesn't exist yet - this is expected"
        return 1
    elif [[ $response == *"error"* ]] || [[ $response == *"Error"* ]]; then
        echo "❌ Error creating table '$table': $response"
        return 1
    else
        echo "✅ Successfully inserted test data into '$table'"
        return 0
    fi
}

echo "📋 Attempting to create tables by inserting test data..."

# Try to create documents table
create_table_by_insert "documents" '{
    "doc_id": "test_doc_123",
    "filename": "test.txt",
    "file_type": "text/plain",
    "file_size": 100,
    "chunks_count": 1,
    "status": "processed"
}' "Create documents table"

# Try to create document_chunks table
create_table_by_insert "document_chunks" '{
    "chunk_id": "test_chunk_123",
    "document_id": "test_doc_123",
    "chunk_index": 0,
    "content": "This is a test chunk"
}' "Create document_chunks table"

# Try to create queries table
create_table_by_insert "queries" '{
    "query_id": "test_query_123",
    "query_text": "What is this about?",
    "response": "This is a test response",
    "processing_time": 1.5
}' "Create queries table"

echo ""
echo "🔍 Testing if tables exist now..."

# Test each table
for table in documents document_chunks queries; do
    test_table "$table"
done

echo ""
echo "🎯 Next Steps:"
echo "If tables still don't exist, you may need to:"
echo "1. Access Supabase dashboard manually"
echo "2. Run the SQL schema in the SQL Editor"
echo "3. Check if your Supabase project has the correct permissions"

echo ""
echo "📄 SQL Schema to run manually:"
echo "Copy and paste this into your Supabase SQL Editor:"
echo ""
cat << 'EOF'
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
EOF

echo ""
echo "🎉 Script completed!"
