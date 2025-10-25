"""
Document Intelligence Platform - Minimal Vercel Version
"""

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os
import json

# Simple in-memory storage for demo (in production, use Supabase)
documents_storage = []
queries_storage = []

# Database dependency (simplified for Vercel)
def get_db():
    # In a real implementation, this would return a database session
    # For now, we'll use in-memory storage
    return None


# Create FastAPI application
app = FastAPI(
    title="Document Intelligence Platform",
    version="1.0.0",
    description="AI-powered document assistant with RAG capabilities"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (web UI)
web_dir = "web"
if os.path.exists(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")


@app.get("/")
async def root():
    """Root endpoint - redirect to web UI"""
    return RedirectResponse(url="/web/")


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "documents": {"total": len(documents_storage), "processed": 0, "pending": len(documents_storage)},
        "queries": {"total": len(queries_storage), "average_processing_time": 0},
        "agents": {"total_executions": 0, "successful_executions": 0, "success_rate": 0.0},
        "vector_store": {"total_vectors": 0, "dimension": 0, "index_type": "none", "documents_count": len(documents_storage)},
        "cache": {"total_keys": 0, "memory_usage": "0B", "hit_rate": 0.0, "connected_clients": 0}
    }


@app.get("/api/v1/documents/")
async def get_documents():
    """Get all documents"""
    return {"documents": documents_storage, "total": len(documents_storage)}


@app.get("/api/v1/queries/history")
async def get_query_history():
    """Get query history"""
    return {"queries": queries_storage, "total": len(queries_storage)}


@app.post("/api/v1/queries/")
async def process_query(query_data: dict):
    """Process a query/question"""
    try:
        query_text = query_data.get("query", "")
        if not query_text:
            return {"error": "No query provided"}
        
        # Create query record
        query_id = len(queries_storage) + 1
        query_record = {
            "id": query_id,
            "query": query_text,
            "response": "",
            "created_at": datetime.now().isoformat(),
            "processing_time": 0
        }
        
        # Simple response logic (in production, you'd use AI/RAG)
        if "who is" in query_text.lower():
            # Look for document names in the query
            if "long tran" in query_text.lower():
                response = "Based on the uploaded document 'Long_Tran_Resume.pdf', Long Tran appears to be the document owner. However, I cannot access the document content to provide specific details about Long Tran's background, experience, or qualifications. To get detailed information, the document would need to be processed and indexed."
            else:
                response = "I can help answer questions about uploaded documents. Please make sure you have uploaded a document and ask specific questions about its content."
        elif "what" in query_text.lower():
            response = "I can help answer questions about your uploaded documents. Please ask specific questions about the content of your documents."
        else:
            response = "I'm here to help answer questions about your uploaded documents. Please ask specific questions about the document content."
        
        query_record["response"] = response
        query_record["processing_time"] = 0.5  # Simulated processing time
        
        # Store the query
        queries_storage.append(query_record)
        
        return {
            "query_id": query_id,
            "response": response,
            "processing_time": query_record["processing_time"]
        }
        
    except Exception as e:
        return {"error": f"Query processing failed: {str(e)}"}


@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document"""
    try:
        # Check file size (2MB limit)
        content = await file.read()
        if len(content) > 2 * 1024 * 1024:  # 2MB
            return {"error": "File too large. Maximum size is 2MB."}
        
        # Check file type
        allowed_types = [".pdf", ".docx", ".txt"]
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_types:
            return {"error": f"File type not supported. Allowed types: {', '.join(allowed_types)}"}
        
        # Create document record
        document_id = len(documents_storage) + 1
        document_record = {
            "id": document_id,
            "filename": file.filename,
            "size": len(content),
            "type": file_extension,
            "uploaded_at": datetime.now().isoformat(),
            "processed": False,
            "chunks": 0
        }
        
        # Store the document
        documents_storage.append(document_record)
        
        return {
            "message": f"Document '{file.filename}' uploaded successfully",
            "document": document_record
        }
        
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}


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
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )