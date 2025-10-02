"""
Document Intelligence Platform - Supabase Version
Optimized for Vercel deployment with Supabase database
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
from supabase import create_client, Client
import PyPDF2
from docx import Document
import io

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
        self.supabase_url = os.getenv("SUPABASE_URL", "").strip()
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.app_name = os.getenv("APP_NAME", "Document Intelligence Platform").strip()
        self.app_version = os.getenv("APP_VERSION", "1.0.0").strip()
        self.debug = os.getenv("DEBUG", "false").strip().lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip()

settings = Settings()

# Initialize Supabase client
supabase: Optional[Client] = None
if settings.supabase_url and settings.supabase_key:
    try:
        supabase = create_client(settings.supabase_url, settings.supabase_key)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        supabase = None
else:
    print("⚠️ Supabase credentials not found")

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

# In-memory stores for embeddings and metadata (fallback)
embeddings_store: Dict[str, List[float]] = {}
metadata_store: Dict[str, Dict[str, Any]] = {}
documents_db: Dict[str, Dict[str, Any]] = {}

def init_supabase_tables():
    """Initialize Supabase tables if they don't exist"""
    if not supabase:
        print("❌ Supabase client not available")
        return False
    
    try:
        # Create documents table
        supabase.rpc('create_documents_table').execute()
        print("✅ Documents table created/verified")
        
        # Create chunks table  
        supabase.rpc('create_chunks_table').execute()
        print("✅ Chunks table created/verified")
        
        # Create queries table
        supabase.rpc('create_queries_table').execute()
        print("✅ Queries table created/verified")
        
        return True
    except Exception as e:
        print(f"⚠️ Error initializing tables (they might already exist): {e}")
        return True  # Continue anyway

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
    
    # Store in Supabase
    if supabase:
        try:
            # Insert document record
            doc_data = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "file_size": len(file_content),
                "chunks_count": len(chunks),
                "status": "processed",
                "metadata": {
                    "total_chunks": len(chunks),
                    "total_characters": len(text),
                    "processing_time": time.time()
                }
            }
            
            result = supabase.table("documents").insert(doc_data).execute()
            print(f"✅ Document stored in Supabase: {doc_id}")
            
            # Process chunks
            chunk_records = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                embedding = generate_embedding(chunk)
                
                chunk_record = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i
                    }
                }
                chunk_records.append(chunk_record)
            
            # Batch insert chunks
            if chunk_records:
                supabase.table("document_chunks").insert(chunk_records).execute()
                print(f"✅ {len(chunk_records)} chunks stored in Supabase")
            
            return doc_data
            
        except Exception as e:
            print(f"❌ Error storing in Supabase: {e}")
            # Fallback to in-memory storage
            pass
    
    # Fallback: Store in memory
    print("⚠️ Using in-memory storage as fallback")
    doc_data = {
        "id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "file_size": len(file_content),
        "status": "processed",
        "chunks_count": len(chunks),
        "created_at": datetime.now().isoformat(),
        "metadata": {
            "total_chunks": len(chunks),
            "total_characters": len(text),
            "processing_time": time.time()
        }
    }
    
    documents_db[doc_id] = doc_data
    
    # Store chunks and embeddings in memory
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = generate_embedding(chunk)
        
        embeddings_store[chunk_id] = embedding
        metadata_store[chunk_id] = {
            "content": chunk,
            "document_id": doc_id,
            "filename": filename,
            "chunk_index": i
        }
    
    return doc_data

def search_similar_chunks(query: str, k: int = 5) -> List[DocumentChunk]:
    """Search for similar chunks using vector similarity"""
    print(f"Searching for similar chunks for query: {query[:50]}...")
    
    if supabase:
        try:
            # Generate query embedding
            query_embedding = generate_embedding(query)
            
            if not query_embedding or all(x == 0.0 for x in query_embedding):
                print("Query embedding generation failed")
                return []
            
            # Use Supabase vector search (if pgvector is enabled)
            # For now, we'll do a simple text search
            result = supabase.table("document_chunks").select("*").ilike("content", f"%{query}%").limit(k).execute()
            
            chunks = []
            for row in result.data:
                chunks.append(DocumentChunk(
                    id=row["chunk_id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    score=0.8  # Placeholder score
                ))
            
            print(f"Found {len(chunks)} chunks from Supabase")
            return chunks
            
        except Exception as e:
            print(f"Error searching Supabase: {e}")
            # Fallback to in-memory search
            pass
    
    # Fallback: In-memory vector search
    if not embeddings_store:
        print("No embeddings store available")
        return []
    
    if not client:
        print("OpenAI client not available")
        return []
    
    try:
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        if not query_embedding or all(x == 0.0 for x in query_embedding):
            print("Query embedding generation failed")
            return []
        
        print(f"Searching {len(embeddings_store)} chunks for query: {query[:50]}...")
        
        # Calculate similarities
        similarities = []
        for chunk_id, chunk_embedding in embeddings_store.items():
            if not chunk_embedding or all(x == 0.0 for x in chunk_embedding):
                continue
                
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk_id, similarity))
        
        if not similarities:
            print("No valid similarities found")
            return []
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = similarities[:k]
        
        print(f"Top similarities: {[(chunk_id, f'{score:.3f}') for chunk_id, score in top_chunks[:3]]}")
        
        results = []
        for chunk_id, score in top_chunks:
            if chunk_id in metadata_store:
                metadata = metadata_store[chunk_id]
                results.append(DocumentChunk(
                    id=chunk_id,
                    content=metadata["content"],
                    metadata={
                        "document_id": metadata["document_id"],
                        "filename": metadata["filename"],
                        "chunk_index": metadata["chunk_index"]
                    },
                    score=score
                ))
        
        print(f"Returning {len(results)} chunks from vector search")
        return results
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
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

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so and provide what information you can."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on document context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
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
        
        # Perform similarity search
        search_results = search_similar_chunks(query_text, k=k*2)
        print(f"Found {len(search_results)} similar chunks")
        
        # Take top k results
        top_chunks = search_results[:k]
        
        # Generate response using LLM
        response = generate_response(query_text, top_chunks)
        
        # Generate query ID
        query_id = f"query_{int(time.time())}"
        
        # Store query in Supabase
        if supabase:
            try:
                query_data = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "response": response,
                    "processing_time": time.time() - start_time
                }
                supabase.table("queries").insert(query_data).execute()
                print(f"✅ Query stored in Supabase: {query_id}")
            except Exception as e:
                print(f"⚠️ Error storing query in Supabase: {e}")
        
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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "supabase_connected": supabase is not None,
        "openai_configured": client is not None
    }

@app.get("/api/v1/debug")
async def debug_info():
    """Debug endpoint to check system state"""
    return {
        "supabase_configured": bool(supabase),
        "supabase_url": settings.supabase_url[:20] + "..." if settings.supabase_url else "Not set",
        "openai_configured": bool(client),
        "openai_key_preview": settings.openai_api_key[:10] + "..." if settings.openai_api_key else "Not set",
        "documents_count": len(documents_db),
        "chunks_count": len(metadata_store),
        "embeddings_count": len(embeddings_store),
        "documents": list(documents_db.keys()),
        "sample_chunk": list(metadata_store.values())[0] if metadata_store else None,
        "sample_embedding_length": len(list(embeddings_store.values())[0]) if embeddings_store else 0
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
        if supabase:
            result = supabase.table("documents").select("*").order("created_at", desc=True).execute()
            documents = []
            for row in result.data:
                documents.append(DocumentResponse(
                    id=row["doc_id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    file_size=row["file_size"],
                    status=row["status"],
                    chunks_count=row["chunks_count"],
                    created_at=row["created_at"],
                    metadata=row["metadata"]
                ))
            return documents
        else:
            # Fallback to in-memory storage
            documents = []
            for doc_id, doc_data in documents_db.items():
                documents.append(DocumentResponse(**doc_data))
            return documents
            
    except Exception as e:
        print(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        if supabase:
            # Delete chunks first
            supabase.table("document_chunks").delete().eq("document_id", document_id).execute()
            
            # Delete document
            supabase.table("documents").delete().eq("doc_id", document_id).execute()
            
            print(f"✅ Document {document_id} deleted from Supabase")
        else:
            # Fallback: delete from in-memory storage
            if document_id in documents_db:
                del documents_db[document_id]
                
                # Delete related chunks
                chunks_to_delete = [chunk_id for chunk_id in metadata_store.keys() if chunk_id.startswith(document_id)]
                for chunk_id in chunks_to_delete:
                    del metadata_store[chunk_id]
                    del embeddings_store[chunk_id]
                
                print(f"✅ Document {document_id} deleted from memory")
        
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
        if supabase:
            result = supabase.table("queries").select("*").order("created_at", desc=True).limit(limit).execute()
            return result.data
        else:
            return {"message": "Query history not available in fallback mode"}
            
    except Exception as e:
        print(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting query history: {str(e)}")

@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        if supabase:
            # Get counts from Supabase
            docs_result = supabase.table("documents").select("id", count="exact").execute()
            chunks_result = supabase.table("document_chunks").select("id", count="exact").execute()
            queries_result = supabase.table("queries").select("id", count="exact").execute()
            
            total_documents = docs_result.count or 0
            total_chunks = chunks_result.count or 0
            total_queries = queries_result.count or 0
            
            # Get average processing time
            avg_result = supabase.table("queries").select("processing_time").execute()
            avg_processing_time = 0.0
            if avg_result.data:
                times = [q["processing_time"] for q in avg_result.data if q["processing_time"]]
                avg_processing_time = sum(times) / len(times) if times else 0.0
            
        else:
            # Fallback to in-memory storage
            total_documents = len(documents_db)
            total_chunks = len(metadata_store)
            total_queries = 0  # Not tracked in memory
            avg_processing_time = 0.0
        
        return StatsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            avg_processing_time=avg_processing_time
        )
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Initialize Supabase tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Starting Document Intelligence Platform...")
    print(f"📊 Supabase URL: {settings.supabase_url[:20]}..." if settings.supabase_url else "Not configured")
    print(f"🤖 OpenAI configured: {bool(client)}")
    
    if supabase:
        init_supabase_tables()
    
    print("✅ Application startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
