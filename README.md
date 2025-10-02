# Document Intelligence Platform
A comprehensive AI-powered document assistant with RAG capabilities, Agent orchestration, and advanced caching strategies.

## Features
- **RAG Implementation**: Retrieval-augmented generation for answering queries across multiple documents
- **LangChain Integration**: Advanced pipelines for query parsing, embedding generation, and vector search
- **Agent Orchestration**: Multi-step workflows for deep research patterns
- **Performance Optimization**: Context re-ranking and caching with Redis + PostgreSQL pgvector
- **Cloud Deployment**: AWS infrastructure with EC2, S3, Lambda
- **Monitoring**: Prometheus/Grafana observability stack

## Quick Start

1. **Setup services** (PostgreSQL, Redis):
```bash
./start.sh
```

2. **Run the application**:
```bash
./run.sh
```

3. **Stop the application**:
```bash
./stop.sh
```

## Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys and database credentials
```

3. Initialize the database:
```bash
python scripts/init_db.py
```

4. Start the application:
```bash
uvicorn app.main:app --reload
```

## Architecture

```
├── app/
│   ├── core/           # Core business logic
│   ├── models/         # Database models
│   ├── services/       # Service layer
│   ├── api/           # API endpoints
│   └── main.py        # FastAPI application
├── scripts/           # Utility scripts
├── tests/            # Test suite
└── docker/           # Container configurations
```

## API Documentation
Once running, visit `http://localhost:8000/docs` for interactive API documentation.
