# Environment Variables for Document Intelligence Platform

## Required Variables

### Supabase Configuration
```bash
# Supabase Project URL (from Supabase Dashboard → Settings → API)
SUPABASE_URL=https://your-project-id.supabase.co

# Supabase Anon Public Key (for client-side operations)
SUPABASE_KEY=your_anon_public_key

# Supabase Service Role Key (for server-side operations)
SUPABASE_SERVICE_KEY=your_service_role_key

# PostgreSQL Connection String (from Supabase Dashboard → Settings → Database)
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres
```

### OpenAI Configuration
```bash
# Your OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# OpenAI Model (optional, defaults to gpt-4-turbo-preview)
OPENAI_MODEL=gpt-4-turbo-preview

# Embedding Model (optional, defaults to text-embedding-3-large)
EMBEDDING_MODEL=text-embedding-3-large
```

### Redis Configuration
```bash
# Redis URL (optional, defaults to localhost)
REDIS_URL=redis://localhost:6379/0
```

## Optional Variables

### AWS Configuration (if using S3)
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
```

### Application Configuration
```bash
# App settings (optional)
APP_NAME=Document Intelligence Platform
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Monitoring (optional)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Vector Store Configuration (optional)
FAISS_INDEX_PATH=./data/faiss_index
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000

# Cache Configuration (optional)
CACHE_TTL=3600
MAX_CACHE_SIZE=1000
```

## Environment File Setup

Create a `.env` file in your project root:

```bash
# Copy this template and fill in your actual values
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_anon_public_key
SUPABASE_SERVICE_KEY=your_service_role_key
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres
OPENAI_API_KEY=sk-your-openai-api-key-here
REDIS_URL=redis://localhost:6379/0
```

## Getting Supabase Credentials

1. **Go to Supabase Dashboard**: https://app.supabase.com
2. **Select your project**
3. **Navigate to Settings → API**
4. **Copy the following**:
   - Project URL → `SUPABASE_URL`
   - Project API keys → `SUPABASE_KEY` (anon public)
   - Project API keys → `SUPABASE_SERVICE_KEY` (service_role)
5. **Navigate to Settings → Database**
6. **Copy the connection string** → `DATABASE_URL`

## Security Notes

- **Never commit** your `.env` file to version control
- **Use different keys** for development and production
- **Rotate keys** regularly for security
- **Use environment-specific** Supabase projects

## Production Deployment

For production deployments:

1. **Set environment variables** in your deployment platform
2. **Use production Supabase project**
3. **Enable Row Level Security** (RLS) policies
4. **Set up proper authentication**
5. **Monitor usage** and costs

## Troubleshooting

### Common Issues:

1. **"Invalid API key"**: Check your Supabase keys are correct
2. **"Connection refused"**: Verify DATABASE_URL format
3. **"Permission denied"**: Ensure you're using service_role key for server operations
4. **"Table doesn't exist"**: Run the migration scripts first

### Testing Connection:

```python
# test_env.py
import os
from app.core.config import settings

print(f"Supabase URL: {settings.supabase_url}")
print(f"Database URL: {settings.database_url[:50]}...")
print(f"OpenAI Key: {settings.openai_api_key[:10]}...")
```
