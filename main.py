"""
Document Intelligence Platform - Simplified Vercel Version
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
import uvicorn
import os

from app.core.config import settings
from app.core.database import init_db, get_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting Document Intelligence Platform...")
    try:
        await init_db()
        print("Application startup complete")
    except Exception as e:
        print(f"Startup error: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered document assistant with RAG capabilities",
    lifespan=lifespan
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
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_dir):
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")


@app.get("/")
async def root():
    """Root endpoint - redirect to web UI"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/")


@app.get("/api/v1/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Simple stats without heavy dependencies
        return {
            "documents": {"total": 0, "processed": 0, "pending": 0},
            "queries": {"total": 0, "average_processing_time": 0},
            "agents": {"total_executions": 0, "successful_executions": 0, "success_rate": 0.0},
            "vector_store": {"total_vectors": 0, "dimension": 0, "index_type": "none", "documents_count": 0},
            "cache": {"total_keys": 0, "memory_usage": "0B", "hit_rate": 0.0, "connected_clients": 0}
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {"error": "Stats unavailable"}


@app.get("/api/v1/documents/")
async def get_documents(db: Session = Depends(get_db)):
    """Get all documents"""
    try:
        return {"documents": [], "total": 0}
    except Exception as e:
        print(f"Documents error: {e}")
        return {"documents": [], "total": 0}


@app.get("/api/v1/queries/history")
async def get_query_history(db: Session = Depends(get_db)):
    """Get query history"""
    try:
        return {"queries": [], "total": 0}
    except Exception as e:
        print(f"Query history error: {e}")
        return {"queries": [], "total": 0}


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Document Intelligence Platform API",
        "version": settings.app_version,
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )