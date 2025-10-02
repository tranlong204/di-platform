#!/usr/bin/env python3
"""
Database initialization script
Creates tables and initial data for the Document Intelligence Platform
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import init_db, engine, Base
from app.models import Document, DocumentChunk, Query, Agent, AgentExecution
from loguru import logger


async def main():
    """Initialize the database"""
    try:
        logger.info("Initializing database...")
        
        # Create all tables
        await init_db()
        
        logger.info("Database initialization completed successfully!")
        
        # Print some useful information
        logger.info("Database tables created:")
        logger.info("- documents")
        logger.info("- document_chunks") 
        logger.info("- queries")
        logger.info("- agents")
        logger.info("- agent_executions")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
