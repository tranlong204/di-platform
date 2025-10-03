"""
Document Intelligence Platform - Render PostgreSQL Version
Optimized for Vercel deployment with Render PostgreSQL database
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
import PyPDF2
from docx import Document
import io
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence Platform",
    description="AI-powered document analysis with RAG capabilities",
    version="1.0.0"
)

# Pydantic models
class DocumentUpload(BaseModel):
    filename: str
    file_type: str
    file_size: int
    status: str = "processed"
    chunks_count: int = 0
    created_at: str
    metadata: Dict[str, Any] = {}

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: float = 0.0

class QueryRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    use_cache: bool = True
    rerank_context: bool = True

class QueryResponse(BaseModel):
    query_id: str
    query_text: str
    response: str
    processing_time: float
    chunks_used: List[DocumentChunk] = []
    cached: bool = False
    created_at: str

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunks_count: int
    created_at: str
    metadata: Dict[str, Any] = {}

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_processing_time: float

# Settings
class Settings:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.app_name = os.getenv("APP_NAME", "Document Intelligence Platform").strip()
        self.app_version = os.getenv("APP_VERSION", "1.0.0").strip()
        self.debug = os.getenv("DEBUG", "false").strip().lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip()

settings = Settings()

# Database connection
def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(settings.database_url)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(50) UNIQUE NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(100) NOT NULL,
                file_size INTEGER NOT NULL,
                status VARCHAR(50) DEFAULT 'processed',
                chunks_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        
        # Create chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(100) UNIQUE NOT NULL,
                document_id VARCHAR(50) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create queries table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_id VARCHAR(50) UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                response TEXT,
                processing_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("Database tables initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

# Initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    if client:
        print("✅ OpenAI client initialized successfully")
    else:
        print("⚠️ OpenAI API key not configured")
except Exception as e:
    print(f"❌ Failed to initialize OpenAI client: {e}")
    client = None

def generate_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI"""
    if not client:
        print("OpenAI client not available - returning dummy embedding")
        return [0.1] * 1536
    
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        print(f"Generated embedding with length {len(embedding)}")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return [0.1] * 1536

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if chunk.strip()]

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

def process_document(file_content: bytes, filename: str, file_type: str) -> Dict[str, Any]:
    """Process uploaded document and extract text"""
    print(f"Processing document: {filename} ({file_type})")
    
    # Extract text based on file type
    if file_type.lower() == 'application/pdf':
        text = extract_text_from_pdf(file_content)
    elif file_type.lower() in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
        text = extract_text_from_docx(file_content)
    else:
        text = file_content.decode('utf-8', errors='ignore')
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text content found in document")
    
    print(f"Extracted {len(text)} characters from {filename}")
    
    # Split text into chunks
    chunks = split_text_into_chunks(text)
    print(f"Split into {len(chunks)} chunks")
    
    # Generate document ID
    doc_id = f"doc_{int(time.time())}"
    
    # Store in PostgreSQL
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cur = conn.cursor()
        
        # Insert document record
        cur.execute("""
            INSERT INTO documents (doc_id, filename, file_type, file_size, chunks_count, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                file_size = EXCLUDED.file_size,
                chunks_count = EXCLUDED.chunks_count,
                metadata = EXCLUDED.metadata
        """, (
            doc_id,
            filename,
            file_type,
            len(file_content),
            len(chunks),
            json.dumps({
                "total_chunks": len(chunks),
                "total_characters": len(text),
                "processing_time": time.time()
            })
        ))
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = generate_embedding(chunk)
            
            # Insert chunk record
            cur.execute("""
                INSERT INTO document_chunks (chunk_id, document_id, chunk_index, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, (
                chunk_id,
                doc_id,
                i,
                chunk,
                json.dumps(embedding),  # Store as JSON string
                json.dumps({
                    "filename": filename,
                    "chunk_index": i
                })
            ))
        
        conn.commit()
        
        # Get the created document
        cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
        doc_record = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return {
            "id": doc_record[1], # doc_id
            "filename": doc_record[2],
            "file_type": doc_record[3],
            "file_size": doc_record[4],
            "status": doc_record[5],
            "chunks_count": doc_record[6],
            "created_at": doc_record[7].isoformat(),
            "metadata": doc_record[8]
        }
    except Exception as e:
        conn.rollback()
        print(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if conn:
            conn.close()

def search_similar_chunks(query: str, k: int = 5) -> List[DocumentChunk]:
    """Search for similar chunks using vector similarity"""
    print(f"Searching for similar chunks for query: {query[:50]}...")
    
    try:
        conn = get_db_connection()
        if not conn:
            print("Database connection failed for search")
            return []
        
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Generate query embedding
            print("Generating query embedding...")
            query_embedding = generate_embedding(query)
            
            if not query_embedding or all(x == 0.0 for x in query_embedding):
                print("Query embedding generation failed")
                return []
            
            print(f"Query embedding generated successfully, length: {len(query_embedding)}")
            
            # Get all chunks with embeddings
            print("Fetching chunks from database...")
            cur.execute("SELECT chunk_id, content, embedding, metadata FROM document_chunks WHERE embedding IS NOT NULL")
            chunks_data = cur.fetchall()
            
            print(f"Found {len(chunks_data)} chunks with embeddings")
            
            if not chunks_data:
                print("No chunks with embeddings found")
                return []
            
            # Calculate similarities
            similarities = []
            for chunk_data in chunks_data:
                try:
                    chunk_embedding = json.loads(chunk_data['embedding'])
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    print(f"Chunk {chunk_data['chunk_id']}: similarity = {similarity:.4f}")
                    similarities.append((chunk_data, similarity))
                except Exception as e:
                    print(f"Error processing chunk {chunk_data['chunk_id']}: {e}")
                    continue
            
            print(f"Similarities list length: {len(similarities)}")
            print(f"Similarities list type: {type(similarities)}")
            
            if not similarities:
                print("No valid similarities found")
                return []
            
            # Sort by similarity and get top k
            print(f"Total similarities calculated: {len(similarities)}")
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_chunks = similarities[:k]
            print(f"Top {k} chunks selected: {len(top_chunks)}")
            
            print(f"Top similarities: {[(chunk['chunk_id'], f'{score:.3f}') for chunk, score in top_chunks[:3]]}")
            
            results = []
            for chunk_data, score in top_chunks:
                try:
                    print(f"Processing chunk {chunk_data['chunk_id']} with score {score:.3f}")
                    print(f"Raw metadata: {chunk_data['metadata']}")
                    print(f"Content preview: {chunk_data['content'][:100]}...")
                    
                    # Handle metadata - it might be a dict or JSON string
                    if isinstance(chunk_data['metadata'], dict):
                        metadata = chunk_data['metadata']
                    else:
                        metadata = json.loads(chunk_data['metadata']) if chunk_data['metadata'] else {}
                    
                    print(f"Parsed metadata: {metadata}")
                    
                    chunk_obj = DocumentChunk(
                        id=chunk_data['chunk_id'],
                        content=chunk_data['content'],
                        metadata={
                            "document_id": metadata.get("document_id"),
                            "filename": metadata.get("filename"),
                            "chunk_index": metadata.get("chunk_index")
                        },
                        score=score
                    )
                    results.append(chunk_obj)
                    print(f"Successfully created DocumentChunk for {chunk_data['chunk_id']}")
                except Exception as e:
                    print(f"Error creating DocumentChunk for {chunk_data['chunk_id']}: {e}")
                    continue
            
            print(f"Returning {len(results)} chunks from vector search")
            return results
            
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []
        finally:
            if conn:
                conn.close()
                
    except Exception as e:
        print(f"CRITICAL ERROR in search_similar_chunks: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return []

def generate_response(query: str, context_chunks: List[DocumentChunk]) -> str:
    """Generate AI response using OpenAI"""
    if not client:
        # Fallback response when OpenAI is not available
        if context_chunks:
            top_chunk = context_chunks[0]
            return f"Based on the document '{top_chunk.metadata['filename']}', here's what I found:\n\n{top_chunk.content[:500]}..."
        else:
            return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    # Prepare context
    context_text = "\n\n".join([chunk.content for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided document context.

Context from documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. Format your response clearly with bullet points or numbered lists when appropriate. Keep responses concise but informative. If the context doesn't contain enough information to answer the question, please say so and provide what information you can."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on document context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        print(f"Debug: AI response length: {len(ai_response)}")
        print(f"Debug: AI response preview: {ai_response[:100]}...")
        
        # If AI response is too generic, provide fallback
        if len(ai_response) < 50 or "I don't know" in ai_response.lower() or "I cannot" in ai_response.lower():
            print("AI response seems inadequate, providing fallback")
            if context_chunks:
                top_chunk = context_chunks[0]
                return f"Based on the document '{top_chunk.metadata['filename']}', here's what I found:\n\n{top_chunk.content[:500]}..."
        
        return ai_response
        
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        # Fallback to simple text extraction
        if context_chunks:
            top_chunk = context_chunks[0]
            return f"Based on the document '{top_chunk.metadata['filename']}', here's what I found:\n\n{top_chunk.content[:500]}..."
        else:
            return f"Error generating response: {str(e)}"

def process_query(query_text: str, k: int = 5) -> Dict[str, Any]:
    """Process a query using RAG"""
    start_time = time.time()
    
    try:
        print(f"Processing query: {query_text[:50]}...")
        print(f"Debug: Query length: {len(query_text)}")
        
        # Perform similarity search
        print("Starting similarity search...")
        search_results = search_similar_chunks(query_text, k=k*2)
        print(f"Similarity search completed. Found {len(search_results)} similar chunks")
        
        # Take top k results
        top_chunks = search_results[:k]
        
        # Generate response using LLM
        response = generate_response(query_text, top_chunks)
        
        # Generate query ID
        query_id = f"query_{int(time.time())}"
        
        # Store query in PostgreSQL
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO queries (query_id, query_text, response, processing_time)
                    VALUES (%s, %s, %s, %s)
                """, (
                    query_id,
                    query_text,
                    response,
                    time.time() - start_time
                ))
                conn.commit()
                cur.close()
                conn.close()
                print(f"✅ Query stored in PostgreSQL: {query_id}")
            except Exception as e:
                print(f"⚠️ Error storing query in PostgreSQL: {e}")
        
        processing_time = time.time() - start_time
        
        return {
            "query_id": query_id,
            "query_text": query_text,
            "response": response,
            "processing_time": processing_time,
            "chunks_used": [chunk.dict() for chunk in top_chunks],
            "cached": False,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# API Routes
@app.get("/api/v1/init-db")
async def init_database_endpoint():
    """Initialize database tables"""
    try:
        if init_database():
            return {"message": "Database tables initialized successfully", "status": "success"}
        else:
            return {"message": "Database initialization failed", "status": "error"}
    except Exception as e:
        return {"message": f"Database initialization error: {str(e)}", "status": "error"}

@app.get("/web")
async def web_ui():
    """Web UI endpoint"""
    try:
        with open("web/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"message": "Web UI not found", "status": "error"}

@app.get("/")
async def root():
    """Root endpoint - serve web UI"""
    try:
        with open("web/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"message": "Document Intelligence Platform API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    conn = get_db_connection()
    db_connected = conn is not None
    if conn:
        conn.close()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_connected,
        "openai_configured": client is not None
    }

@app.get("/api/v1/debug-embeddings")
async def debug_embeddings():
    """Debug endpoint to check embeddings"""
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed"}
    
    try:
        cur = conn.cursor()
        
        # Check if we have chunks with embeddings
        cur.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL")
        chunks_with_embeddings = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cur.fetchone()[0]
        
        # Get a sample chunk
        cur.execute("SELECT chunk_id, content, embedding FROM document_chunks LIMIT 1")
        sample_chunk = cur.fetchone()
        
        # Test embedding generation
        test_embedding = generate_embedding("test query")
        
        return {
            "total_chunks": total_chunks,
            "chunks_with_embeddings": chunks_with_embeddings,
            "sample_chunk_id": sample_chunk[0] if sample_chunk else None,
            "sample_content_preview": sample_chunk[1][:100] if sample_chunk else None,
            "sample_embedding_length": len(json.loads(sample_chunk[2])) if sample_chunk and sample_chunk[2] else None,
            "test_embedding_length": len(test_embedding),
            "test_embedding_preview": test_embedding[:5] if test_embedding else None
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if conn:
            conn.close()

@app.get("/api/v1/debug")
async def debug_info():
    """Debug endpoint to check system state"""
    conn = get_db_connection()
    if not conn:
        return {
            "database_connected": False,
            "database_url": settings.database_url[:20] + "..." if settings.database_url else "Not set",
            "openai_configured": bool(client),
            "openai_key_preview": settings.openai_api_key[:10] + "..." if settings.openai_api_key else "Not set"
        }
    
    try:
        cur = conn.cursor()
        
        # Get counts
        cur.execute("SELECT COUNT(*) FROM documents")
        docs_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        chunks_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM queries")
        queries_count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return {
            "database_connected": True,
            "database_url": settings.database_url[:20] + "..." if settings.database_url else "Not set",
            "openai_configured": bool(client),
            "openai_key_preview": settings.openai_api_key[:10] + "..." if settings.openai_api_key else "Not set",
            "documents_count": docs_count,
            "chunks_count": chunks_count,
            "queries_count": queries_count
        }
    except Exception as e:
        return {
            "database_connected": True,
            "database_url": settings.database_url[:20] + "..." if settings.database_url else "Not set",
            "openai_configured": bool(client),
            "openai_key_preview": settings.openai_api_key[:10] + "..." if settings.openai_api_key else "Not set",
            "error": str(e)
        }

@app.post("/api/v1/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        print(f"Uploading document: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        
        # Process document
        doc_data = process_document(file_content, file.filename, file.content_type)
        
        return DocumentResponse(**doc_data)
        
    except Exception as e:
        print(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/v1/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get all uploaded documents"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
        docs_data = cur.fetchall()
        
        documents = []
        for doc_data in docs_data:
            documents.append(DocumentResponse(
                id=doc_data["doc_id"],
                filename=doc_data["filename"],
                file_type=doc_data["file_type"],
                file_size=doc_data["file_size"],
                status=doc_data["status"],
                chunks_count=doc_data["chunks_count"],
                created_at=doc_data["created_at"].isoformat(),
                metadata=doc_data["metadata"]
            ))
        
        cur.close()
        conn.close()
        return documents
        
    except Exception as e:
        print(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cur = conn.cursor()
        
        # Delete chunks first
        cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
        
        # Delete document
        cur.execute("DELETE FROM documents WHERE doc_id = %s", (document_id,))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"✅ Document {document_id} deleted from PostgreSQL")
        return {"message": f"Document {document_id} deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/v1/queries", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """Process a query using RAG"""
    try:
        result = process_query(request.query, request.k)
        return QueryResponse(**result)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/v1/queries/history")
async def get_query_history(limit: int = 20):
    """Get query history"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM queries ORDER BY created_at DESC LIMIT %s", (limit,))
        queries_data = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [dict(query) for query in queries_data]
        
    except Exception as e:
        print(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting query history: {str(e)}")

@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cur = conn.cursor()
        
        # Get counts
        cur.execute("SELECT COUNT(*) FROM documents")
        total_documents = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total_chunks = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM queries")
        total_queries = cur.fetchone()[0]
        
        # Get average processing time
        cur.execute("SELECT AVG(processing_time) FROM queries WHERE processing_time IS NOT NULL")
        avg_result = cur.fetchone()[0]
        avg_processing_time = float(avg_result) if avg_result else 0.0
        
        cur.close()
        conn.close()
        
        return StatsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            avg_processing_time=avg_processing_time
        )
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Starting Document Intelligence Platform...")
    print(f"📊 Database URL: {settings.database_url[:20]}..." if settings.database_url else "Not configured")
    print(f"🤖 OpenAI configured: {bool(client)}")
    
    # Initialize database tables
    if init_database():
        print("✅ Database tables initialized successfully")
    else:
        print("⚠️ Database initialization failed")
    
    print("✅ Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

