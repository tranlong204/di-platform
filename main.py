"""
Vercel entry point for Document Intelligence Platform
"""

from app.main import app

# Export the FastAPI app for Vercel
__all__ = ["app"]
