"""
Database configuration and connection management
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import redis
from supabase import create_client, Client
from app.core.config import settings

# Supabase PostgreSQL Database
engine = create_engine(
    settings.database_url,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Supabase Client
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

# Redis Cache
redis_client = redis.from_url(settings.redis_url, decode_responses=True)


async def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_redis():
    """Redis dependency for FastAPI"""
    return redis_client


async def get_supabase():
    """Supabase dependency for FastAPI"""
    return supabase


async def init_db():
    """Initialize database tables"""
    # Import all models to ensure they are registered
    from app.models import Document, DocumentChunk, Query, Agent, AgentExecution
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
