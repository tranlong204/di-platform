"""
Database models for the Document Intelligence Platform
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class Document(Base):
    """Document model for storing document metadata"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False)
    title = Column(String(255))
    summary = Column(Text)
    metadata_json = Column(JSON)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class DocumentChunk(Base):
    """Document chunk model for storing text chunks"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding_id = Column(String(100))  # Reference to FAISS index
    metadata_json = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Query(Base):
    """Query model for storing user queries and responses"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    query_hash = Column(String(64), nullable=False, index=True)
    context_documents = Column(JSON)  # List of document IDs used
    context_chunks = Column(JSON)  # List of chunk IDs used
    similarity_scores = Column(JSON)  # Similarity scores for retrieved chunks
    processing_time = Column(Float)  # Time taken to process query
    user_feedback = Column(Integer)  # 1-5 rating
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Agent(Base):
    """Agent model for storing agent configurations and workflows"""
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    workflow_config = Column(JSON, nullable=False)
    tools = Column(JSON)  # Available tools for the agent
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AgentExecution(Base):
    """Agent execution model for tracking agent runs"""
    __tablename__ = "agent_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, nullable=False, index=True)
    query_id = Column(Integer, nullable=False, index=True)
    execution_steps = Column(JSON)  # Steps taken during execution
    final_result = Column(Text)
    execution_time = Column(Float)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
