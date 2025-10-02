"""
Document Intelligence Platform - Minimal Vercel Version
Using FastAPI with minimal dependencies
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Document Intelligence Platform",
    description="AI-powered document analysis with RAG capabilities",
    version="1.0.0"
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
    context_documents: List[str]
    context_chunks: List[str]
    similarity_scores: List[float]
    processing_time: float
    cached: bool

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    status: str
    created_at: str

class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    avg_processing_time: float

# Root endpoint - redirect to web UI
@app.get("/")
async def root():
    """Root endpoint - redirect to web UI"""
    return RedirectResponse(url="/web/")

# API information endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Document Intelligence Platform API",
        "version": "1.0.0",
        "status": "running",
        "deployment": "Vercel Minimal",
        "endpoints": {
            "web_ui": "/web/",
            "api_docs": "/docs",
            "health": "/health",
            "queries": "/api/v1/queries/",
            "documents": "/api/v1/documents/",
            "stats": "/api/v1/stats/"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "deployment": "Vercel Minimal",
        "timestamp": "2024-10-02T18:00:00Z"
    }

# Query processing endpoint
@app.post("/api/v1/queries/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using RAG"""
    try:
        # Simple response logic for demo
        if 'platform' in request.query.lower():
            response_text = "This is a Document Intelligence Platform that uses AI to analyze and answer questions about uploaded documents. It features RAG (Retrieval-Augmented Generation) capabilities for intelligent document processing. This minimal version is deployed on Vercel for maximum reliability and performance."
        elif 'document' in request.query.lower():
            response_text = "You can upload documents in various formats (PDF, DOCX, TXT) and the platform will process them to enable intelligent question-answering. The system uses vector embeddings and similarity search to find relevant information."
        elif 'upload' in request.query.lower():
            response_text = "To upload documents, use the document management section in the web interface. Supported formats include PDF, Word documents, and text files. The platform will automatically process and index your documents for intelligent querying."
        elif 'vercel' in request.query.lower():
            response_text = "This application is deployed on Vercel, which provides excellent performance, automatic scaling, and global CDN distribution. Vercel ensures high availability and fast response times worldwide."
        else:
            response_text = f"I understand you're asking about: {request.query}. This Document Intelligence Platform can help analyze documents and answer questions about their content. It's powered by AI and deployed on Vercel for optimal performance."
        
        return QueryResponse(
            query_id=1,
            query=request.query,
            response=response_text,
            context_documents=[],
            context_chunks=[],
            similarity_scores=[],
            processing_time=0.1,
            cached=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Documents endpoint
@app.get("/api/v1/documents/", response_model=List[DocumentResponse])
async def get_documents():
    """Get list of documents"""
    # Mock data for demo
    return [
        DocumentResponse(
            id=1,
            filename="sample_document.pdf",
            file_type="application/pdf",
            file_size=1024000,
            status="processed",
            created_at="2024-10-02T18:00:00Z"
        ),
        DocumentResponse(
            id=2,
            filename="example.docx",
            file_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size=512000,
            status="processed",
            created_at="2024-10-02T17:30:00Z"
        )
    ]

# Stats endpoint
@app.get("/api/v1/stats/", response_model=StatsResponse)
async def get_stats():
    """Get platform statistics"""
    return StatsResponse(
        total_documents=2,
        total_chunks=150,
        total_queries=25,
        avg_processing_time=0.15
    )

# Web UI endpoint
@app.get("/web/", response_class=HTMLResponse)
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
                max-width: 800px;
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
            .chat-container {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                min-height: 400px;
                text-align: left;
                max-height: 500px;
                overflow-y: auto;
            }
            .message {
                background: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .user-message {
                background: #e3f2fd;
                margin-left: 20px;
            }
            .ai-message {
                background: #f1f8e9;
                margin-right: 20px;
            }
            input {
                width: 100%;
                padding: 15px;
                border: 2px solid #e9ecef;
                border-radius: 25px;
                font-size: 16px;
                margin: 10px 0;
            }
            .sidebar {
                display: flex;
                gap: 20px;
                margin-top: 20px;
            }
            .sidebar > div {
                flex: 1;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 15px;
            }
            .sidebar h3 {
                color: #333;
                margin-bottom: 15px;
            }
            .document-item {
                background: white;
                padding: 10px;
                border-radius: 8px;
                margin: 5px 0;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Document Intelligence Platform</h1>
            <p>AI-powered document analysis with RAG capabilities</p>
            <div class="status">✅ Successfully deployed on Vercel!</div>
            <p>This is a minimal version running on Vercel with FastAPI for maximum reliability and performance.</p>
            
            <div class="chat-container" id="chatContainer">
                <div class="message ai-message">
                    👋 Hello! I'm your Document Intelligence assistant. I can help you understand documents and answer questions about their content. Try asking me about the platform!
                </div>
            </div>
            
            <input type="text" id="queryInput" placeholder="Ask me anything about documents or the platform..." />
            <button class="btn" onclick="sendQuery()">Send Query</button>
            
            <div class="sidebar">
                <div>
                    <h3>📚 Your Documents</h3>
                    <div id="documentsList">
                        <div class="document-item">📄 sample_document.pdf (1.0 MB)</div>
                        <div class="document-item">📄 example.docx (512 KB)</div>
                    </div>
                </div>
                <div>
                    <h3>📊 Platform Stats</h3>
                    <div id="statsInfo">
                        <div class="document-item">📄 2 Documents</div>
                        <div class="document-item">🔍 150 Chunks</div>
                        <div class="document-item">❓ 25 Queries</div>
                        <div class="document-item">⚡ 0.15s Avg Time</div>
                    </div>
                </div>
            </div>
            
            <br><br>
            <a href="/api" class="btn">📚 API Info</a>
            <a href="/health" class="btn">❤️ Health Check</a>
            <a href="/docs" class="btn">📖 API Docs</a>
        </div>
        
        <script>
            async function sendQuery() {
                const input = document.getElementById('queryInput');
                const query = input.value.trim();
                
                if (!query) return;
                
                // Add user message
                const chatContainer = document.getElementById('chatContainer');
                const userMessage = document.createElement('div');
                userMessage.className = 'message user-message';
                userMessage.innerHTML = '<strong>You:</strong> ' + query;
                chatContainer.appendChild(userMessage);
                
                input.value = '';
                
                // Show thinking indicator
                const thinkingMessage = document.createElement('div');
                thinkingMessage.className = 'message ai-message';
                thinkingMessage.innerHTML = '<strong>AI:</strong> 🤔 Thinking...';
                chatContainer.appendChild(thinkingMessage);
                
                try {
                    const response = await fetch('/api/v1/queries/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: query,
                            k: 3,
                            use_cache: false,
                            rerank_context: true
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Remove thinking message
                    chatContainer.removeChild(thinkingMessage);
                    
                    // Add AI response
                    const aiMessage = document.createElement('div');
                    aiMessage.className = 'message ai-message';
                    aiMessage.innerHTML = '<strong>AI:</strong> ' + data.response;
                    chatContainer.appendChild(aiMessage);
                    
                    // Scroll to bottom
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                } catch (error) {
                    console.error('Error:', error);
                    // Remove thinking message
                    chatContainer.removeChild(thinkingMessage);
                    
                    const errorMessage = document.createElement('div');
                    errorMessage.className = 'message ai-message';
                    errorMessage.innerHTML = '<strong>Error:</strong> Sorry, I encountered an error processing your request.';
                    chatContainer.appendChild(errorMessage);
                }
            }
            
            // Handle Enter key
            document.getElementById('queryInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendQuery();
                }
            });
            
            // Load documents and stats on page load
            async function loadData() {
                try {
                    const [documentsResponse, statsResponse] = await Promise.all([
                        fetch('/api/v1/documents/'),
                        fetch('/api/v1/stats/')
                    ]);
                    
                    const documents = await documentsResponse.json();
                    const stats = await statsResponse.json();
                    
                    // Update documents list
                    const documentsList = document.getElementById('documentsList');
                    documentsList.innerHTML = documents.map(doc => 
                        `<div class="document-item">📄 ${doc.filename} (${(doc.file_size / 1024).toFixed(0)} KB)</div>`
                    ).join('');
                    
                    // Update stats
                    const statsInfo = document.getElementById('statsInfo');
                    statsInfo.innerHTML = `
                        <div class="document-item">📄 ${stats.total_documents} Documents</div>
                        <div class="document-item">🔍 ${stats.total_chunks} Chunks</div>
                        <div class="document-item">❓ ${stats.total_queries} Queries</div>
                        <div class="document-item">⚡ ${stats.avg_processing_time}s Avg Time</div>
                    `;
                    
                } catch (error) {
                    console.error('Error loading data:', error);
                }
            }
            
            // Load data when page loads
            loadData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Mount static files if they exist
try:
    from fastapi.staticfiles import StaticFiles
    import os
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    if os.path.exists(web_dir):
        app.mount("/web-static", StaticFiles(directory=web_dir, html=True), name="web-static")
except ImportError:
    pass
