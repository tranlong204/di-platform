-- Supabase Migration: Add vector search capabilities
-- This migration adds pgvector extension and embedding columns for semantic search

-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding columns to document_chunks table
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS embedding VECTOR(1536); -- OpenAI text-embedding-3-large dimensions

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create function for semantic search
CREATE OR REPLACE FUNCTION search_documents_by_embedding(
    query_embedding VECTOR(1536),
    match_threshold FLOAT DEFAULT 0.5,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_id INTEGER,
    content TEXT,
    similarity FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.content,
        1 - (dc.embedding <=> query_embedding) as similarity,
        dc.metadata_json as metadata
    FROM document_chunks dc
    WHERE dc.embedding IS NOT NULL
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to update document chunk embeddings
CREATE OR REPLACE FUNCTION update_chunk_embedding(
    chunk_id INTEGER,
    new_embedding VECTOR(1536)
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE document_chunks 
    SET embedding = new_embedding 
    WHERE id = chunk_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to get similar chunks
CREATE OR REPLACE FUNCTION get_similar_chunks(
    target_chunk_id INTEGER,
    similarity_threshold FLOAT DEFAULT 0.7,
    max_results INTEGER DEFAULT 5
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_id INTEGER,
    content TEXT,
    similarity FLOAT
) AS $$
DECLARE
    target_embedding VECTOR(1536);
BEGIN
    -- Get the embedding of the target chunk
    SELECT embedding INTO target_embedding
    FROM document_chunks 
    WHERE id = target_chunk_id;
    
    -- Return if no embedding found
    IF target_embedding IS NULL THEN
        RETURN;
    END IF;
    
    -- Find similar chunks
    RETURN QUERY
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.content,
        1 - (dc.embedding <=> target_embedding) as similarity
    FROM document_chunks dc
    WHERE dc.id != target_chunk_id 
    AND dc.embedding IS NOT NULL
    AND 1 - (dc.embedding <=> target_embedding) >= similarity_threshold
    ORDER BY dc.embedding <=> target_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
