"""
Context re-ranking service for improving RAG accuracy
"""

import numpy as np
from typing import List, Dict, Any
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ContextReranker:
    """Service for re-ranking retrieved context to improve relevance"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    async def rerank(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank search results based on multiple factors"""
        try:
            if len(search_results) <= 1:
                return search_results
            
            # Extract texts for analysis
            query_text = query.lower()
            context_texts = [result['content'].lower() for result in search_results]
            
            # Calculate various relevance scores
            scores = []
            for i, result in enumerate(search_results):
                context_text = context_texts[i]
                
                # 1. Keyword overlap score
                keyword_score = self._calculate_keyword_overlap(query_text, context_text)
                
                # 2. Semantic similarity score (using TF-IDF)
                semantic_score = self._calculate_semantic_similarity(query_text, context_text)
                
                # 3. Position-based score (prefer earlier chunks)
                position_score = self._calculate_position_score(result['chunk_index'])
                
                # 4. Length penalty (prefer moderate length chunks)
                length_score = self._calculate_length_score(len(context_text))
                
                # 5. Original similarity score
                original_score = result['similarity_score']
                
                # Combined score with weights
                combined_score = (
                    0.3 * keyword_score +
                    0.25 * semantic_score +
                    0.15 * position_score +
                    0.1 * length_score +
                    0.2 * original_score
                )
                
                scores.append(combined_score)
            
            # Sort results by combined score
            scored_results = list(zip(search_results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Update similarity scores with reranked scores
            reranked_results = []
            for result, score in scored_results:
                result['similarity_score'] = score
                reranked_results.append(result)
            
            logger.info(f"Re-ranked {len(search_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in context re-ranking: {str(e)}")
            return search_results
    
    def _calculate_keyword_overlap(self, query: str, context: str) -> float:
        """Calculate keyword overlap score"""
        try:
            # Extract keywords from query
            query_words = set(re.findall(r'\b\w+\b', query))
            context_words = set(re.findall(r'\b\w+\b', context))
            
            if not query_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = query_words.intersection(context_words)
            union = query_words.union(context_words)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating keyword overlap: {str(e)}")
            return 0.0
    
    def _calculate_semantic_similarity(self, query: str, context: str) -> float:
        """Calculate semantic similarity using TF-IDF"""
        try:
            # Fit TF-IDF on both texts
            texts = [query, context]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def _calculate_position_score(self, chunk_index: int) -> float:
        """Calculate position-based score (prefer earlier chunks)"""
        # Normalize chunk index to 0-1 range, with earlier chunks getting higher scores
        # This is a simple linear decay
        max_chunks = 100  # Assume max 100 chunks per document
        normalized_index = min(chunk_index / max_chunks, 1.0)
        return 1.0 - normalized_index
    
    def _calculate_length_score(self, text_length: int) -> float:
        """Calculate length-based score (prefer moderate length)"""
        # Prefer chunks between 200-800 characters
        optimal_min, optimal_max = 200, 800
        
        if optimal_min <= text_length <= optimal_max:
            return 1.0
        elif text_length < optimal_min:
            # Shorter chunks get lower scores
            return text_length / optimal_min
        else:
            # Longer chunks get lower scores
            return optimal_max / text_length
    
    async def rerank_with_diversity(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-rank with diversity to avoid redundant results"""
        try:
            if len(search_results) <= max_results:
                return await self.rerank(query, search_results)
            
            # First, rerank all results
            reranked = await self.rerank(query, search_results)
            
            # Then apply diversity filtering
            diverse_results = []
            used_documents = set()
            
            for result in reranked:
                doc_id = result['document_id']
                
                # Allow multiple chunks from same document but limit
                doc_count = sum(1 for r in diverse_results if r['document_id'] == doc_id)
                
                if doc_count < 2:  # Max 2 chunks per document
                    diverse_results.append(result)
                    used_documents.add(doc_id)
                
                if len(diverse_results) >= max_results:
                    break
            
            logger.info(f"Applied diversity filtering: {len(diverse_results)} results")
            return diverse_results
            
        except Exception as e:
            logger.error(f"Error in diversity re-ranking: {str(e)}")
            return await self.rerank(query, search_results)
