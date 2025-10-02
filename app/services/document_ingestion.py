"""
Document ingestion service for managing document uploads and processing
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session
from loguru import logger

from app.models import Document, DocumentChunk
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStoreService


class DocumentIngestionService:
    """Service for ingesting and processing documents"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStoreService()
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
    
    async def ingest_document(
        self, 
        file_path: str, 
        db: Session,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Ingest a document into the system"""
        try:
            # Process the document
            processed_doc = self.processor.process_document(file_path)
            
            # Check if document already exists
            existing_doc = db.query(Document).filter(
                Document.content_hash == processed_doc['metadata']['content_hash']
            ).first()
            
            if existing_doc:
                logger.info(f"Document already exists: {existing_doc.filename}")
                return {
                    'document_id': existing_doc.id,
                    'status': 'already_exists',
                    'message': 'Document already processed'
                }
            
            # Save document metadata to database
            document = Document(
                filename=processed_doc['metadata']['filename'],
                file_path=file_path,
                file_type=processed_doc['file_type'],
                file_size=processed_doc['metadata']['file_size'],
                content_hash=processed_doc['metadata']['content_hash'],
                metadata=metadata or {}
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Process chunks
            chunk_ids = await self._process_chunks(
                document.id, 
                processed_doc['chunks'], 
                db
            )
            
            # Generate embeddings and add to vector store
            await self._add_to_vector_store(document.id, processed_doc['chunks'])
            
            # Mark document as processed
            document.processed = True
            db.commit()
            
            logger.info(f"Successfully ingested document: {document.filename}")
            
            return {
                'document_id': document.id,
                'status': 'success',
                'chunks_processed': len(chunk_ids),
                'message': 'Document processed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            raise
    
    async def _process_chunks(
        self, 
        document_id: int, 
        chunks: List[Dict[str, Any]], 
        db: Session
    ) -> List[int]:
        """Process document chunks and save to database"""
        chunk_ids = []
        
        for chunk_data in chunks:
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                content_hash=chunk_data['content_hash'],
                token_count=chunk_data['token_count'],
                metadata={
                    'start_word': chunk_data['start_word'],
                    'end_word': chunk_data['end_word']
                }
            )
            
            db.add(chunk)
            db.commit()
            db.refresh(chunk)
            chunk_ids.append(chunk.id)
        
        return chunk_ids
    
    async def _add_to_vector_store(
        self, 
        document_id: int, 
        chunks: List[Dict[str, Any]]
    ):
        """Add document chunks to vector store"""
        try:
            await self.vector_store.add_documents(document_id, chunks)
            logger.info(f"Added {len(chunks)} chunks to vector store for document {document_id}")
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            raise
    
    async def get_document_by_id(self, document_id: int, db: Session) -> Optional[Document]:
        """Get document by ID"""
        return db.query(Document).filter(Document.id == document_id).first()
    
    async def get_document_chunks(self, document_id: int, db: Session) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        return db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    async def delete_document(self, document_id: int, db: Session) -> bool:
        """Delete a document and its associated data"""
        try:
            # Get document
            document = await self.get_document_by_id(document_id, db)
            if not document:
                return False
            
            # Delete chunks from vector store
            await self.vector_store.delete_document(document_id)
            
            # Delete chunks from database
            db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).delete()
            
            # Delete document
            db.delete(document)
            db.commit()
            
            # Delete file if it exists
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            logger.info(f"Deleted document: {document.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            db.rollback()
            return False
    
    async def list_documents(self, db: Session, limit: int = 100) -> List[Document]:
        """List all documents"""
        return db.query(Document).limit(limit).all()
