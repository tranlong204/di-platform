"""
API routes for document management
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from loguru import logger

from app.core.database import get_db
from app.services.document_ingestion import DocumentIngestionService
from app.models import Document, DocumentChunk
from pydantic import BaseModel


router = APIRouter()


class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    processed: bool
    created_at: str


class DocumentUploadResponse(BaseModel):
    document_id: int
    status: str
    message: str
    chunks_processed: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Save uploaded file
        import os
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            import json
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, ignoring")
        
        # Process document
        ingestion_service = DocumentIngestionService()
        result = await ingestion_service.ingest_document(file_path, db, doc_metadata)
        
        return DocumentUploadResponse(**result)
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all documents"""
    try:
        ingestion_service = DocumentIngestionService()
        documents = await ingestion_service.list_documents(db, limit)
        
        document_responses = [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                processed=doc.processed,
                created_at=doc.created_at.isoformat()
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total=len(document_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get document by ID"""
    try:
        ingestion_service = DocumentIngestionService()
        document = await ingestion_service.get_document_by_id(document_id, db)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            file_size=document.file_size,
            processed=document.processed,
            created_at=document.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get all chunks for a document"""
    try:
        ingestion_service = DocumentIngestionService()
        chunks = await ingestion_service.get_document_chunks(document_id, db)
        
        return {
            "document_id": document_id,
            "chunks": [
                {
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "created_at": chunk.created_at.isoformat()
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error getting document chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document and its associated data"""
    try:
        ingestion_service = DocumentIngestionService()
        success = await ingestion_service.delete_document(document_id, db)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
