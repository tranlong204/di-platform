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
import PyPDF2
from docx import Document
from io import BytesIO
from supabase import create_client, Client
from pydantic_settings import BaseSettings

# Supabase Configuration
class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str
    supabase_service_key: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()

# Initialize Supabase client
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

# Simple in-memory storage for demo (in production, use Supabase)
documents_storage = []
queries_storage = []

# Database dependency (simplified for Vercel)
def get_db():
    # In a real implementation, this would return a database session
    # For now, we'll use in-memory storage
    return None


def extract_text_from_document(content: bytes, file_extension: str) -> str:
    """Extract text from uploaded document"""
    try:
        if file_extension == ".pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        elif file_extension == ".docx":
            # Extract text from DOCX
            doc = Document(BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        
        elif file_extension == ".txt":
            # Extract text from TXT
            return content.decode('utf-8').strip()
        
        else:
            return ""
    
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


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
    processed_docs = sum(1 for doc in documents_storage if doc.get("processed", False))
    total_chunks = sum(doc.get("chunks", 0) for doc in documents_storage)
    avg_processing_time = sum(q.get("processing_time", 0) for q in queries_storage) / len(queries_storage) if queries_storage else 0
    
    return {
        "documents": {"total": len(documents_storage), "processed": processed_docs, "pending": len(documents_storage) - processed_docs},
        "queries": {"total": len(queries_storage), "average_processing_time": avg_processing_time},
        "agents": {"total_executions": 0, "successful_executions": 0, "success_rate": 0.0},
        "vector_store": {"total_vectors": total_chunks, "dimension": 0, "index_type": "none", "documents_count": len(documents_storage)},
        "cache": {"total_keys": 0, "memory_usage": "0B", "hit_rate": 0.0, "connected_clients": 0}
    }


@app.get("/api/v1/documents/")
async def get_documents():
    """Get all documents"""
    try:
        # Try to fetch from Supabase first
        supabase_response = supabase.table("documents").select("*").execute()
        if supabase_response.data:
            return {"documents": supabase_response.data, "total": len(supabase_response.data)}
    except Exception as e:
        print(f"Supabase fetch failed: {e}")
    
    # Fallback to memory storage
    return {"documents": documents_storage, "total": len(documents_storage)}


@app.get("/api/v1/queries/history")
async def get_query_history():
    """Get query history"""
    try:
        # Try to fetch from Supabase first
        supabase_response = supabase.table("queries").select("*").order("created_at", desc=True).execute()
        if supabase_response.data:
            return {"queries": supabase_response.data, "total": len(supabase_response.data)}
    except Exception as e:
        print(f"Supabase fetch failed: {e}")
    
    # Fallback to memory storage
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
        response = "I'm here to help answer questions about your uploaded documents."
        
        # Search through uploaded documents for relevant content
        if documents_storage:
            for doc in documents_storage:
                if doc.get("full_content"):
                    content = doc["full_content"].lower()
                    query_lower = query_text.lower()
                    
                    # Look for specific information in the document
                    if "who is" in query_lower and "long tran" in query_lower:
                        # Extract relevant information about Long Tran from the document
                        lines = doc["full_content"].split('\n')
                        relevant_info = []
                        
                        for line in lines:
                            line_lower = line.lower()
                            if any(keyword in line_lower for keyword in ['long tran', 'experience', 'education', 'skills', 'summary', 'objective']):
                                relevant_info.append(line.strip())
                        
                        if relevant_info:
                            response = f"Based on the document '{doc['filename']}', here's what I found about Long Tran:\n\n" + "\n".join(relevant_info[:5])  # Show first 5 relevant lines
                        else:
                            response = f"I found the document '{doc['filename']}' but couldn't extract specific information about Long Tran. The document contains: {doc['content'][:200]}..."
                    
                    elif "what" in query_lower or "tell me" in query_lower:
                        # General information extraction
                        response = f"Based on the document '{doc['filename']}', here's some information:\n\n{doc['content'][:500]}..."
                        break
                    
                    elif any(keyword in query_lower for keyword in ['experience', 'education', 'skills', 'work', 'job']):
                        # Look for specific sections
                        lines = doc["full_content"].split('\n')
                        relevant_lines = []
                        
                        for line in lines:
                            line_lower = line.lower()
                            if any(keyword in line_lower for keyword in ['experience', 'education', 'skills', 'work', 'job', 'position', 'degree']):
                                relevant_lines.append(line.strip())
                        
                        if relevant_lines:
                            response = f"Here's relevant information from '{doc['filename']}':\n\n" + "\n".join(relevant_lines[:10])
                        else:
                            response = f"I found the document '{doc['filename']}' but couldn't find specific information about that topic. The document contains: {doc['content'][:300]}..."
                        break
        else:
            response = "I don't see any uploaded documents. Please upload a document first, then ask questions about its content."
        
        query_record["response"] = response
        query_record["processing_time"] = 0.5  # Simulated processing time
        
        # Store the query in Supabase
        try:
            # Save to Supabase queries table
            supabase_response = supabase.table("queries").insert({
                "query": query_text,
                "response": response,
                "created_at": datetime.now().isoformat(),
                "processing_time": query_record["processing_time"]
            }).execute()
            
            # Also store in memory for immediate access
            queries_storage.append(query_record)
            
            return {
                "query_id": query_id,
                "response": response,
                "processing_time": query_record["processing_time"],
                "supabase_id": supabase_response.data[0]["id"] if supabase_response.data else None
            }
        except Exception as db_error:
            # Fallback to memory storage if Supabase fails
            queries_storage.append(query_record)
            return {
                "query_id": query_id,
                "response": response,
                "processing_time": query_record["processing_time"],
                "warning": f"Database save failed: {str(db_error)}"
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
        
        # Extract text from document
        extracted_text = extract_text_from_document(content, file_extension)
        
        document_record = {
            "id": document_id,
            "filename": file.filename,
            "size": len(content),
            "type": file_extension,
            "uploaded_at": datetime.now().isoformat(),
            "processed": True if extracted_text else False,
            "chunks": 1 if extracted_text else 0,
            "content": extracted_text[:1000] if extracted_text else "",  # Store first 1000 chars for demo
            "full_content": extracted_text  # Store full content for querying
        }
        
        # Store the document in Supabase
        try:
            # Save to Supabase documents table
            supabase_response = supabase.table("documents").insert({
                "filename": file.filename,
                "file_path": f"/uploads/{file.filename}",  # Virtual path
                "file_type": file_extension,
                "file_size": len(content),
                "uploaded_at": datetime.now().isoformat(),
                "processed": True if extracted_text else False,
                "content": extracted_text[:1000] if extracted_text else "",
                "full_content": extracted_text
            }).execute()
            
            # Also store in memory for immediate access
            documents_storage.append(document_record)
            
            return {
                "message": f"Document '{file.filename}' uploaded successfully",
                "document": document_record,
                "supabase_id": supabase_response.data[0]["id"] if supabase_response.data else None
            }
        except Exception as db_error:
            # Fallback to memory storage if Supabase fails
            documents_storage.append(document_record)
            return {
                "message": f"Document '{file.filename}' uploaded successfully (memory only)",
                "document": document_record,
                "warning": f"Database save failed: {str(db_error)}"
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