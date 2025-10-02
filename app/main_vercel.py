"""
Document Intelligence Platform - Vercel Optimized Version
"""
import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence Platform",
    version="1.0.0",
    description="AI-powered document analysis with RAG capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    k: int = 5
    use_cache: bool = True
    rerank_context: bool = True

class QueryResponse(BaseModel):
    query_id: int
    query: str
    response: str
    context_documents: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    similarity_scores: List[float]
    processing_time: float
    cached: bool = False

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: str
    metadata_json: Optional[Dict[str, Any]] = None

# Global variables for simplified functionality
documents_db = []
queries_db = []
query_counter = 0

@app.get("/")
async def root():
    """Root endpoint - redirect to web UI"""
    return {"message": "Document Intelligence Platform API", "version": "1.0.0", "status": "running"}

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Document Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "web_ui": "/web/",
            "api_docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/api/v1/queries/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using simplified RAG"""
    global query_counter
    
    try:
        query_counter += 1
        
        # Simulate processing time
        import time
        start_time = time.time()
        
        # Simple response based on query content
        if "platform" in request.query.lower():
            response = "This is a Document Intelligence Platform that uses AI to analyze and answer questions about uploaded documents. It features RAG (Retrieval-Augmented Generation) capabilities for intelligent document processing."
        elif "document" in request.query.lower():
            response = "You can upload documents in various formats (PDF, DOCX, TXT) and the platform will process them to enable intelligent question-answering."
        elif "upload" in request.query.lower():
            response = "To upload documents, use the document management section in the web interface. Supported formats include PDF, Word documents, and text files."
        else:
            response = f"I understand you're asking about: {request.query}. This platform can help analyze documents and answer questions about their content."
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query_id=query_counter,
            query=request.query,
            response=response,
            context_documents=[],
            context_chunks=[],
            similarity_scores=[],
            processing_time=processing_time,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/queries/history")
async def get_query_history(limit: int = 50):
    """Get query history"""
    return {
        "queries": queries_db[-limit:] if queries_db else [],
        "total": len(queries_db)
    }

@app.post("/api/v1/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents"""
    try:
        uploaded_docs = []
        
        for file in files:
            if file.filename:
                # Simulate document processing
                doc_id = len(documents_db) + 1
                doc = {
                    "id": doc_id,
                    "filename": file.filename,
                    "file_type": file.content_type or "unknown",
                    "file_size": 1024,  # Simulated size
                    "status": "processed",
                    "created_at": "2025-01-01T00:00:00Z",
                    "metadata_json": {"pages": 1, "chunks": 1}
                }
                documents_db.append(doc)
                uploaded_docs.append(doc)
        
        return {
            "message": f"Successfully uploaded {len(uploaded_docs)} documents",
            "documents": uploaded_docs
        }
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/documents/")
async def get_documents():
    """Get all documents"""
    return {
        "documents": documents_db,
        "total": len(documents_db)
    }

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    global documents_db
    
    # Find and remove document
    documents_db = [doc for doc in documents_db if doc["id"] != document_id]
    
    return {"message": f"Document {document_id} deleted successfully"}

@app.get("/api/v1/stats")
async def get_stats():
    """Get platform statistics"""
    return {
        "total_documents": len(documents_db),
        "total_queries": len(queries_db),
        "platform_status": "running"
    }

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
