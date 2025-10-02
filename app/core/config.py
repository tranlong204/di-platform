"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-large"
    
    # Database Configuration
    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None
    
    # Application Configuration
    app_name: str = "Document Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Monitoring Configuration
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Vector Store Configuration
    faiss_index_path: str = "./data/faiss_index"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4000
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
