"""
RAG query processing service using LangChain
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.cache_service import CacheService
from app.services.context_reranker import ContextReranker
from app.models import Query, Document


class RAGQueryService:
    """Service for processing queries using RAG with LangChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=settings.max_tokens
        )
        self.vector_store = VectorStoreService()
        self.cache_service = CacheService()
        self.context_reranker = ContextReranker()
        
        # Setup prompts
        self.system_prompt = self._create_system_prompt()
        self.qa_prompt = self._create_qa_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the RAG system"""
        return """You are an AI assistant specialized in answering questions based on document content. 
        You have access to a knowledge base of documents and should provide accurate, comprehensive answers 
        based on the retrieved context.
        
        Guidelines:
        1. Always base your answers on the provided context
        2. If the context doesn't contain enough information, say so clearly
        3. Cite specific parts of the documents when relevant
        4. Provide detailed explanations when appropriate
        5. If asked about multiple documents, synthesize information across them
        6. Maintain a professional and helpful tone
        """
    
    def _create_qa_prompt(self) -> ChatPromptTemplate:
        """Create QA prompt template"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            """Context from documents:
            {context}
            
            Question: {question}
            
            Please provide a comprehensive answer based on the context above."""
        )
        
        return ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])
    
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
                cached_result = await self.cache_service.get_query_result(query_text)
                if cached_result:
                    logger.info(f"Returning cached result for query: {query_text[:50]}...")
                    return cached_result
            
            # Generate query hash for tracking
            query_hash = self._generate_query_hash(query_text)
            
            # Perform similarity search
            search_results = await self.vector_store.similarity_search(query_text, k=k*2)
            
            if not search_results:
                # Still create a query record even with no results
                query_hash = self._generate_query_hash(query_text)
                response_text = "I couldn't find any relevant information in the document collection."
                
                query_record = Query(
                    query_text=query_text,
                    response_text=response_text,
                    query_hash=query_hash,
                    context_documents=[],
                    context_chunks=[],
                    similarity_scores=[],
                    processing_time=time.time() - start_time
                )
                
                db.add(query_record)
                db.commit()
                db.refresh(query_record)
                
                return {
                    'query_id': query_record.id,
                    'query': query_text,
                    'response': response_text,
                    'context_documents': [],
                    'context_chunks': [],
                    'similarity_scores': [],
                    'processing_time': time.time() - start_time,
                    'cached': False
                }
            
            # Rerank context if enabled
            if rerank_context:
                search_results = await self.context_reranker.rerank(query_text, search_results)
            
            # Take top k results
            top_results = search_results[:k]
            
            # Prepare context
            context = self._prepare_context(top_results)
            context_documents = list(set([r['document_id'] for r in top_results]))
            context_chunks = [r['vector_id'] for r in top_results]
            similarity_scores = [r['similarity_score'] for r in top_results]
            
            # Generate response using LLM
            response = await self._generate_response(query_text, context)
            
            processing_time = time.time() - start_time
            
            # Save query to database
            query_record = Query(
                query_text=query_text,
                response_text=response,
                query_hash=query_hash,
                context_documents=context_documents,
                context_chunks=context_chunks,
                similarity_scores=similarity_scores,
                processing_time=processing_time
            )
            
            db.add(query_record)
            db.commit()
            db.refresh(query_record)
            
            result = {
                'query_id': query_record.id,
                'query': query_text,
                'response': response,
                'context_documents': context_documents,
                'context_chunks': context_chunks,
                'similarity_scores': similarity_scores,
                'processing_time': processing_time,
                'cached': False
            }
            
            # Cache the result
            if use_cache:
                await self.cache_service.cache_query_result(query_text, result)
            
            logger.info(f"Processed query in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _generate_query_hash(self, query_text: str) -> str:
        """Generate hash for query tracking"""
        import hashlib
        return hashlib.sha256(query_text.encode()).hexdigest()
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Document {result['document_id']}, Chunk {result['chunk_index']}:\n"
                f"{result['content']}\n"
                f"Similarity Score: {result['similarity_score']:.3f}\n"
            )
        
        return "\n".join(context_parts)
    
    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM"""
        try:
            messages = self.qa_prompt.format_messages(
                context=context,
                question=query
            )
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def process_multi_document_query(
        self, 
        query_text: str, 
        document_ids: List[int], 
        db: Session,
        k: int = 5
    ) -> Dict[str, Any]:
        """Process query across specific documents"""
        try:
            # Perform similarity search with document filter
            search_results = await self.vector_store.similarity_search(
                query_text, 
                k=k*2, 
                filter_document_ids=document_ids
            )
            
            if not search_results:
                return {
                    'query': query_text,
                    'response': f"No relevant information found in the specified documents.",
                    'context_documents': document_ids,
                    'context_chunks': [],
                    'similarity_scores': [],
                    'processing_time': 0,
                    'cached': False
                }
            
            # Rerank and process
            search_results = await self.context_reranker.rerank(query_text, search_results)
            top_results = search_results[:k]
            
            # Generate response
            context = self._prepare_context(top_results)
            response = await self._generate_response(query_text, context)
            
            return {
                'query': query_text,
                'response': response,
                'context_documents': document_ids,
                'context_chunks': [r['vector_id'] for r in top_results],
                'similarity_scores': [r['similarity_score'] for r in top_results],
                'processing_time': 0,
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"Error processing multi-document query: {str(e)}")
            raise
    
    async def get_query_history(self, db: Session, limit: int = 50) -> List[Query]:
        """Get query history"""
        return db.query(Query).order_by(Query.created_at.desc()).limit(limit).all()
    
    async def update_query_feedback(self, query_id: int, feedback: int, db: Session) -> bool:
        """Update user feedback for a query"""
        try:
            query = db.query(Query).filter(Query.id == query_id).first()
            if query:
                query.user_feedback = feedback
                db.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating query feedback: {str(e)}")
            db.rollback()
            return False
