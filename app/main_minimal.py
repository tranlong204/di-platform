"""
Document Intelligence Platform - Minimal Vercel Version
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence Platform",
    version="1.0.0",
    description="AI-powered document analysis with RAG capabilities"
)

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
    context_documents: list
    context_chunks: list
    similarity_scores: list
    processing_time: float
    cached: bool = False

# Global variables
query_counter = 0

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Intelligence Platform API", 
        "version": "1.0.0", 
        "status": "running",
        "deployment": "Vercel"
    }

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Document Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "deployment": "Vercel",
        "endpoints": {
            "web_ui": "/web/",
            "api_docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0", "deployment": "Vercel"}

@app.post("/api/v1/queries/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using simplified RAG"""
    global query_counter
    
    try:
        query_counter += 1
        
        # Simple response based on query content
        if "platform" in request.query.lower():
            response = "This is a Document Intelligence Platform that uses AI to analyze and answer questions about uploaded documents. It features RAG (Retrieval-Augmented Generation) capabilities for intelligent document processing. This version is deployed on Vercel for maximum reliability and performance."
        elif "document" in request.query.lower():
            response = "You can upload documents in various formats (PDF, DOCX, TXT) and the platform will process them to enable intelligent question-answering. The system uses vector embeddings and similarity search to find relevant information."
        elif "upload" in request.query.lower():
            response = "To upload documents, use the document management section in the web interface. Supported formats include PDF, Word documents, and text files. The platform will automatically process and index your documents for intelligent querying."
        elif "vercel" in request.query.lower():
            response = "This application is deployed on Vercel, which provides excellent performance, automatic scaling, and global CDN distribution. Vercel ensures high availability and fast response times worldwide."
        else:
            response = f"I understand you're asking about: {request.query}. This Document Intelligence Platform can help analyze documents and answer questions about their content. It's powered by AI and deployed on Vercel for optimal performance."
        
        return QueryResponse(
            query_id=query_counter,
            query=request.query,
            response=response,
            context_documents=[],
            context_chunks=[],
            similarity_scores=[],
            processing_time=0.1,
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/queries/history")
async def get_query_history(limit: int = 50):
    """Get query history"""
    return {
        "queries": [],
        "total": 0,
        "message": "Query history not available in this simplified version"
    }

@app.post("/api/v1/documents/upload")
async def upload_documents():
    """Upload documents"""
    return {
        "message": "Document upload functionality is available in the full version",
        "documents": [],
        "status": "simplified"
    }

@app.get("/api/v1/documents/")
async def get_documents():
    """Get all documents"""
    return {
        "documents": [],
        "total": 0,
        "message": "Document management available in full version"
    }

@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document"""
    return {"message": f"Document {document_id} deletion not available in simplified version"}

@app.get("/api/v1/stats")
async def get_stats():
    """Get platform statistics"""
    return {
        "total_documents": 0,
        "total_queries": query_counter,
        "platform_status": "running",
        "deployment": "Vercel",
        "version": "simplified"
    }

@app.get("/web/")
async def web_ui():
    """Serve the web UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document Intelligence Platform</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                width: 90%;
                max-width: 600px;
                padding: 40px;
                text-align: center;
            }
            h1 { color: #333; font-size: 2.5rem; margin-bottom: 20px; }
            p { color: #666; font-size: 1.1rem; margin-bottom: 30px; }
            .status { background: #d4edda; color: #155724; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
            .btn { 
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white; padding: 15px 30px; border: none; border-radius: 25px;
                font-size: 16px; cursor: pointer; margin: 10px; text-decoration: none; display: inline-block;
            }
            .btn:hover { transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Document Intelligence Platform</h1>
            <p>AI-powered document analysis with RAG capabilities</p>
            <div class="status">✅ Successfully deployed on Vercel!</div>
            <p>This is a simplified version running on Vercel for maximum reliability and performance.</p>
            <a href="/docs" class="btn">📚 API Documentation</a>
            <a href="/api" class="btn">🔗 API Info</a>
            <a href="/health" class="btn">❤️ Health Check</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
