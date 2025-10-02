"""
FAISS vector store service for document embeddings and similarity search
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import tiktoken
from loguru import logger

from app.core.config import settings
from langchain_openai import OpenAIEmbeddings


class VectorStoreService:
    """Service for managing FAISS vector store and embeddings"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        # Configure embeddings client; OpenAIEmbeddings uses sync HTTP under the hood,
        # so we guard calls with try/except and log durations to detect hangs.
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        self.index_path = Path(settings.faiss_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index lazily to avoid blocking startup
        self.dimension = 3072  # OpenAI text-embedding-3-large dimension
        self.index = None  # Will be loaded on first use
        self.id_to_metadata = None  # Will be loaded on first use
        self._index_loaded = False
        
        # Tokenizer for counting tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Mark as initialized
        self._initialized = True
    
    def _ensure_index_loaded(self):
        """Ensure FAISS index and metadata are loaded (lazy loading)"""
        if not self._index_loaded:
            logger.info("Loading FAISS index on first use...")
            self.index = self._load_or_create_index()
            self.id_to_metadata = self._load_metadata()
            self._index_loaded = True
            logger.info("FAISS index loaded successfully")
    
    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        """Load existing FAISS index or create new one"""
        index_file = self.index_path / "faiss_index.bin"
        
        if index_file.exists():
            logger.info("Loading existing FAISS index")
            return faiss.read_index(str(index_file))
        else:
            logger.info("Creating new FAISS index")
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            return index
    
    def _load_metadata(self) -> Dict[int, Dict[str, Any]]:
        """Load metadata mapping"""
        metadata_file = self.index_path / "metadata.pkl"
        
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    
    def _save_index(self):
        """Save FAISS index to disk"""
        index_file = self.index_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))
        
        metadata_file = self.index_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)
    
    async def add_documents(self, document_id: int, chunks: List[Dict[str, Any]]):
        """Add document chunks to vector store"""
        try:
            # Ensure index is loaded
            self._ensure_index_loaded()
            
            # Generate embeddings for all chunks
            texts = [chunk['content'] for chunk in chunks]
            embeddings = await self._generate_embeddings(texts)
            
            # Add to FAISS index
            start_id = self.index.ntotal
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata for each chunk
            for i, chunk in enumerate(chunks):
                vector_id = start_id + i
                self.id_to_metadata[vector_id] = {
                    'document_id': document_id,
                    'chunk_index': chunk['chunk_index'],
                    'content_hash': chunk['content_hash'],
                    'content': chunk['content'],
                    'token_count': chunk['token_count']
                }
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            start_time = None
            import time as _time
            start_time = _time.time()
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Embeddings generated in {_time.time() - start_time:.2f}s")
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            # Ensure index is loaded
            self._ensure_index_loaded()
            
            # Check if index is empty
            if self.index.ntotal == 0:
                logger.info("FAISS index is empty, returning empty results")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(query)
            
            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k * 2  # Get more results for filtering
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                metadata = self.id_to_metadata.get(idx)
                if not metadata:
                    continue
                
                # Apply document filter if specified
                if filter_document_ids and metadata['document_id'] not in filter_document_ids:
                    continue
                
                results.append({
                    'vector_id': int(idx),
                    'document_id': metadata['document_id'],
                    'chunk_index': metadata['chunk_index'],
                    'content': metadata['content'],
                    'content_hash': metadata['content_hash'],
                    'similarity_score': float(score),
                    'token_count': metadata['token_count']
                })
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    async def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        try:
            logger.info("Generating query embedding...")
            import time as _time
            _t0 = _time.time()
            embedding = self.embeddings.embed_query(query)
            logger.info(f"Query embedding generated in {_time.time() - _t0:.2f}s")
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    async def delete_document(self, document_id: int):
        """Delete all vectors for a document"""
        try:
            # Ensure index is loaded
            self._ensure_index_loaded()
            
            # Find all vector IDs for this document
            vector_ids_to_remove = []
            for vector_id, metadata in self.id_to_metadata.items():
                if metadata['document_id'] == document_id:
                    vector_ids_to_remove.append(vector_id)
            
            if not vector_ids_to_remove:
                logger.info(f"No vectors found for document {document_id}")
                return
            
            # Remove from metadata
            for vector_id in vector_ids_to_remove:
                del self.id_to_metadata[vector_id]
            
            # Rebuild index without deleted vectors
            await self._rebuild_index()
            
            logger.info(f"Deleted {len(vector_ids_to_remove)} vectors for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {str(e)}")
            raise
    
    async def _rebuild_index(self):
        """Rebuild FAISS index from remaining metadata"""
        try:
            # Ensure index is loaded
            self._ensure_index_loaded()
            
            # Create new index
            new_index = faiss.IndexFlatIP(self.dimension)
            
            # Get all remaining embeddings
            remaining_metadata = list(self.id_to_metadata.items())
            if not remaining_metadata:
                self.index = new_index
                self._save_index()
                return
            
            # Extract embeddings and rebuild
            vector_ids = [item[0] for item in remaining_metadata]
            embeddings = []
            
            for vector_id in vector_ids:
                # Get embedding from original index
                embedding = self.index.reconstruct(vector_id)
                embeddings.append(embedding)
            
            # Add to new index
            embeddings_array = np.array(embeddings).astype('float32')
            new_index.add(embeddings_array)
            
            # Update metadata with new vector IDs
            new_metadata = {}
            for i, (old_vector_id, metadata) in enumerate(remaining_metadata):
                new_metadata[i] = metadata
            
            self.index = new_index
            self.id_to_metadata = new_metadata
            
            # Save updated index
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        # Ensure index is loaded
        self._ensure_index_loaded()
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP',
            'documents_count': len(set(meta['document_id'] for meta in self.id_to_metadata.values()))
        }
