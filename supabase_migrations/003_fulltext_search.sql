-- Supabase Migration: Add full-text search capabilities
-- This migration adds full-text search indexes and functions

-- Add full-text search columns
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS search_vector tsvector;

ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- Create function to update document search vectors
CREATE OR REPLACE FUNCTION update_document_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.summary, '')), 'B') ||
        setweight(to_tsvector('english', COALESCE(NEW.filename, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create function to update chunk search vectors
CREATE OR REPLACE FUNCTION update_chunk_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to automatically update search vectors
CREATE TRIGGER update_documents_search_vector
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_document_search_vector();

CREATE TRIGGER update_chunks_search_vector
    BEFORE INSERT OR UPDATE ON document_chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_search_vector();

-- Create GIN indexes for full-text search
CREATE INDEX IF NOT EXISTS idx_documents_search_vector 
ON documents USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector 
ON document_chunks USING GIN (search_vector);

-- Create function for full-text search in documents
CREATE OR REPLACE FUNCTION search_documents_text(
    search_query TEXT,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    document_id INTEGER,
    filename VARCHAR(255),
    title VARCHAR(255),
    summary TEXT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id as document_id,
        d.filename,
        d.title,
        d.summary,
        ts_rank(d.search_vector, plainto_tsquery('english', search_query)) as rank
    FROM documents d
    WHERE d.search_vector @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function for full-text search in chunks
CREATE OR REPLACE FUNCTION search_chunks_text(
    search_query TEXT,
    limit_count INTEGER DEFAULT 20
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_id INTEGER,
    content TEXT,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id as chunk_id,
        dc.document_id,
        dc.content,
        ts_rank(dc.search_vector, plainto_tsquery('english', search_query)) as rank
    FROM document_chunks dc
    WHERE dc.search_vector @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create hybrid search function (combines vector and text search)
CREATE OR REPLACE FUNCTION hybrid_search(
    search_query TEXT,
    query_embedding VECTOR(1536),
    vector_weight FLOAT DEFAULT 0.7,
    text_weight FLOAT DEFAULT 0.3,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id INTEGER,
    document_id INTEGER,
    content TEXT,
    vector_similarity FLOAT,
    text_rank REAL,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            1 - (dc.embedding <=> query_embedding) as vector_similarity,
            0::REAL as text_rank
        FROM document_chunks dc
        WHERE dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> query_embedding
        LIMIT limit_count * 2
    ),
    text_results AS (
        SELECT 
            dc.id as chunk_id,
            dc.document_id,
            dc.content,
            0::FLOAT as vector_similarity,
            ts_rank(dc.search_vector, plainto_tsquery('english', search_query)) as text_rank
        FROM document_chunks dc
        WHERE dc.search_vector @@ plainto_tsquery('english', search_query)
        ORDER BY text_rank DESC
        LIMIT limit_count * 2
    ),
    combined_results AS (
        SELECT * FROM vector_results
        UNION
        SELECT * FROM text_results
    )
    SELECT 
        cr.chunk_id,
        cr.document_id,
        cr.content,
        cr.vector_similarity,
        cr.text_rank,
        (cr.vector_similarity * vector_weight + cr.text_rank * text_weight) as combined_score
    FROM combined_results cr
    ORDER BY combined_score DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
