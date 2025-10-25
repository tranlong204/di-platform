"""
RAG query processing service using OpenAI directly
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from openai import OpenAI
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import CacheService
from app.services.context_reranker import ContextReranker
from app.models import Query, Document


class RAGQueryService:
    """Service for processing queries using RAG with OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.vector_store = VectorStoreService()
        self.cache_service = CacheService()
        self.reranker = ContextReranker()
    
    async def process_query(
        self, 
        query_text: str, 
        db: Session,
        k: int = 5,
        use_cache: bool = True,
        rerank_context: bool = True
    ) -> Dict[str, Any]:
        """Process a query using RAG"""
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_result = await self.cache_service.get_cached_result(query_text)
                if cached_result:
                    logger.info(f"Returning cached result for query: {query_text}")
                    return cached_result
            
            # Retrieve relevant documents
            relevant_docs = await self.vector_store.search_similar_documents(
                query_text, k=k
            )
            
            if not relevant_docs:
                return {
                    'query_id': None,
                    'query': query_text,
                    'response': "I couldn't find any relevant documents to answer your question.",
                    'context_documents': [],
                    'context_chunks': [],
                    'similarity_scores': [],
                    'processing_time': time.time() - start_time,
                    'cached': False
                }
            
            # Rerank context if requested
            if rerank_context and len(relevant_docs) > 1:
                relevant_docs = await self.reranker.rerank_context(
                    query_text, relevant_docs
                )
            
            # Prepare context
            context_text = self._prepare_context(relevant_docs)
            
            # Generate response using OpenAI
            response = await self._generate_response(query_text, context_text)
            
            # Save query to database
            query_record = Query(
                query_text=query_text,
                response_text=response,
                query_hash=self._hash_query(query_text),
                context_documents=[doc['document_id'] for doc in relevant_docs],
                context_chunks=[doc['chunk_id'] for doc in relevant_docs],
                similarity_scores=[doc['similarity'] for doc in relevant_docs],
                processing_time=time.time() - start_time
            )
            
            db.add(query_record)
            db.commit()
            db.refresh(query_record)
            
            # Cache the result
            if use_cache:
                await self.cache_service.cache_result(query_text, {
                    'query_id': query_record.id,
                    'query': query_text,
                    'response': response,
                    'context_documents': [doc['document_id'] for doc in relevant_docs],
                    'context_chunks': [doc['chunk_id'] for doc in relevant_docs],
                    'similarity_scores': [doc['similarity'] for doc in relevant_docs],
                    'processing_time': time.time() - start_time,
                    'cached': True
                })
            
            return {
                'query_id': query_record.id,
                'query': query_text,
                'response': response,
                'context_documents': [doc['document_id'] for doc in relevant_docs],
                'context_chunks': [doc['chunk_id'] for doc in relevant_docs],
                'similarity_scores': [doc['similarity'] for doc in relevant_docs],
                'processing_time': time.time() - start_time,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant documents"""
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}\n")
        return "\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI"""
        try:
            system_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents. 
            Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.
            Be concise and accurate in your response."""
            
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        import hashlib
        return hashlib.sha256(query.encode()).hexdigest()
    
    async def get_query_history(self, db: Session, limit: int = 10) -> List[Query]:
        """Get query history"""
        return db.query(Query).order_by(Query.created_at.desc()).limit(limit).all()
    
    async def get_query_by_id(self, query_id: int, db: Session) -> Optional[Query]:
        """Get query by ID"""
        return db.query(Query).filter(Query.id == query_id).first()
    
    async def delete_query(self, query_id: int, db: Session) -> bool:
        """Delete a query"""
        try:
            query = await self.get_query_by_id(query_id, db)
            if query:
                db.delete(query)
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting query: {str(e)}")
            db.rollback()
            return False