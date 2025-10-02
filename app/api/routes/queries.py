"""
API routes for query processing
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger

from app.core.database import get_db
from app.services.rag_service import RAGQueryService
from app.models import Query


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    use_cache: bool = True
    rerank_context: bool = True


class QueryResponse(BaseModel):
    query_id: int
    query: str
    response: str
    context_documents: List[int]
    context_chunks: List[int]
    similarity_scores: List[float]
    processing_time: float
    cached: bool


class MultiDocumentQueryRequest(BaseModel):
    query: str
    document_ids: List[int]
    k: int = 5


class QueryHistoryResponse(BaseModel):
    queries: List[dict]
    total: int


class FeedbackRequest(BaseModel):
    feedback: int  # 1-5 rating


@router.post("/", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """Process a query using RAG"""
    try:
        rag_service = RAGQueryService()
        result = await rag_service.process_query(
            query_text=request.query,
            db=db,
            k=request.k,
            use_cache=request.use_cache,
            rerank_context=request.rerank_context
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-document", response_model=QueryResponse)
async def process_multi_document_query(
    request: MultiDocumentQueryRequest,
    db: Session = Depends(get_db)
):
    """Process query across specific documents"""
    try:
        rag_service = RAGQueryService()
        result = await rag_service.process_multi_document_query(
            query_text=request.query,
            document_ids=request.document_ids,
            db=db,
            k=request.k
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing multi-document query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=QueryHistoryResponse)
async def get_query_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get query history"""
    try:
        rag_service = RAGQueryService()
        queries = await rag_service.get_query_history(db, limit)
        
        query_data = [
            {
                "id": query.id,
                "query_text": query.query_text,
                "response_text": query.response_text,
                "processing_time": query.processing_time,
                "user_feedback": query.user_feedback,
                "created_at": query.created_at.isoformat()
            }
            for query in queries
        ]
        
        return QueryHistoryResponse(
            queries=query_data,
            total=len(query_data)
        )
        
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{query_id}/feedback")
async def submit_query_feedback(
    query_id: int,
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """Submit feedback for a query"""
    try:
        if not 1 <= request.feedback <= 5:
            raise HTTPException(status_code=400, detail="Feedback must be between 1 and 5")
        
        rag_service = RAGQueryService()
        success = await rag_service.update_query_feedback(query_id, request.feedback, db)
        
        if not success:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {"message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{query_id}")
async def get_query_details(
    query_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific query"""
    try:
        query = db.query(Query).filter(Query.id == query_id).first()
        
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "id": query.id,
            "query_text": query.query_text,
            "response_text": query.response_text,
            "context_documents": query.context_documents,
            "context_chunks": query.context_chunks,
            "similarity_scores": query.similarity_scores,
            "processing_time": query.processing_time,
            "user_feedback": query.user_feedback,
            "created_at": query.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
