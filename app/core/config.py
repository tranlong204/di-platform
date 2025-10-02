"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="placeholder-key", alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-large"
    
    # Database Configuration
    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket_name: Optional[str] = Field(default=None, alias="S3_BUCKET")
    
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
        extra = "ignore"


# Global settings instance
settings = Settings()
