#!/bin/bash
# Script to create Supabase tables using Management API

echo "🚀 Creating Supabase tables via Management API..."

# Supabase configuration
SUPABASE_URL="https://pvmaxqesaxplmztyaujj.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"

# Function to execute SQL
execute_sql() {
    local sql="$1"
    local description="$2"
    
    echo "🔧 $description..."
    
    # Try to execute SQL via Supabase Management API
    response=$(curl -s -X POST "https://api.supabase.com/v1/projects/pvmaxqesaxplmztyaujj/sql" \
        -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$sql\"}")
    
    if [[ $? -eq 0 ]]; then
        echo "✅ $description - Success"
        echo "Response: $response"
        return 0
    else
        echo "❌ $description - Failed"
        return 1
    fi
}

# Create tables
echo "📋 Creating database tables..."

# Enable pgvector extension
execute_sql "CREATE EXTENSION IF NOT EXISTS vector;" "Enable pgvector extension"

# Create documents table
execute_sql "CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(50) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(100) NOT NULL,
    file_size INTEGER NOT NULL,
    status VARCHAR(50) DEFAULT 'processed',
    chunks_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);" "Create documents table"

# Create document_chunks table
execute_sql "CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(100) UNIQUE NOT NULL,
    document_id VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);" "Create document_chunks table"

# Create queries table
execute_sql "CREATE TABLE IF NOT EXISTS queries (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(50) UNIQUE NOT NULL,
    query_text TEXT NOT NULL,
    response TEXT,
    processing_time FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);" "Create queries table"

# Create indexes
execute_sql "CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON document_chunks(chunk_id);
CREATE INDEX IF NOT EXISTS idx_queries_query_id ON queries(query_id);
CREATE INDEX IF NOT EXISTS idx_queries_created_at ON queries(created_at);" "Create indexes"

# Enable RLS
execute_sql "ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE queries ENABLE ROW LEVEL SECURITY;" "Enable Row Level Security"

# Create policies
execute_sql "CREATE POLICY \"Service role can do everything\" ON documents
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY \"Service role can do everything\" ON document_chunks
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY \"Service role can do everything\" ON queries
    FOR ALL USING (auth.role() = 'service_role');" "Create security policies"

echo ""
echo "🔍 Testing if tables were created..."

# Test each table
for table in documents document_chunks queries; do
    echo "Testing table: $table"
    response=$(curl -s -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
        -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY" \
        "$SUPABASE_URL/rest/v1/$table?select=*&limit=1")
    
    if [[ $response == *"PGRST205"* ]]; then
        echo "❌ Table '$table' not found"
    else
        echo "✅ Table '$table' exists and accessible"
    fi
done

echo ""
echo "🎉 Table creation script completed!"
