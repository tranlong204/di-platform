# Supabase Migration Guide for Document Intelligence Platform

## Overview
This guide will help you migrate from your current PostgreSQL setup to Supabase, which provides a managed PostgreSQL database with additional features like real-time subscriptions, authentication, and vector search capabilities.

## Prerequisites
1. A Supabase account (sign up at https://supabase.com)
2. A new Supabase project created
3. Your existing environment variables

## Step 1: Create Supabase Project

1. Go to [Supabase Dashboard](https://app.supabase.com)
2. Click "New Project"
3. Choose your organization
4. Enter project details:
   - Name: `di-platform`
   - Database Password: Generate a strong password
   - Region: Choose closest to your users
5. Click "Create new project"

## Step 2: Get Supabase Credentials

1. Go to your project dashboard
2. Navigate to Settings → API
3. Copy the following values:
   - Project URL (`SUPABASE_URL`)
   - Anon public key (`SUPABASE_KEY`)
   - Service role key (`SUPABASE_SERVICE_KEY`) - for server-side operations

## Step 3: Update Environment Variables

Create or update your `.env` file with the following variables:

```bash
# Existing variables (keep these)
OPENAI_API_KEY=your_openai_api_key
REDIS_URL=redis://localhost:6379/0

# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_anon_public_key
SUPABASE_SERVICE_KEY=your_service_role_key

# Database URL (Supabase PostgreSQL connection string)
DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres
```

## Step 4: Run Database Migrations

1. Install the Supabase CLI (optional but recommended):
   ```bash
   npm install -g supabase
   ```

2. Run the migration scripts in order:
   ```bash
   # Connect to your Supabase database and run:
   psql "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres" -f supabase_migrations/001_initial_schema.sql
   psql "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres" -f supabase_migrations/002_vector_search.sql
   psql "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-ID].supabase.co:5432/postgres" -f supabase_migrations/003_fulltext_search.sql
   ```

   Or use the Supabase Dashboard SQL Editor:
   - Go to SQL Editor in your Supabase dashboard
   - Copy and paste each migration file content
   - Run them in order (001, 002, 003)

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 6: Test the Connection

Create a simple test script to verify everything works:

```python
# test_supabase.py
import asyncio
from app.core.database import supabase, get_supabase

async def test_connection():
    # Test Supabase client
    try:
        result = supabase.table('documents').select('*').limit(1).execute()
        print("✅ Supabase connection successful")
        print(f"Found {len(result.data)} documents")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
```

## Step 7: Update Your Application Code

### Using Supabase Client Directly

For new features, you can use the Supabase client directly:

```python
from app.core.database import supabase

# Insert a document
result = supabase.table('documents').insert({
    'filename': 'test.pdf',
    'file_path': '/path/to/file',
    'file_type': 'pdf',
    'file_size': 1024,
    'content_hash': 'abc123',
    'processed': False
}).execute()

# Query documents
documents = supabase.table('documents').select('*').eq('processed', True).execute()

# Update document
supabase.table('documents').update({'processed': True}).eq('id', 1).execute()
```

### Using SQLAlchemy (Existing Code)

Your existing SQLAlchemy code will continue to work unchanged since Supabase provides a standard PostgreSQL connection.

## Step 8: Enable Vector Search (Optional)

If you want to use Supabase's built-in vector search capabilities:

1. Go to Database → Extensions in your Supabase dashboard
2. Enable the `vector` extension
3. The migration scripts already include vector search setup

## Step 9: Set Up Authentication (Optional)

Supabase provides built-in authentication. To enable it:

1. Go to Authentication → Settings in your Supabase dashboard
2. Configure your authentication providers
3. Update your frontend to use Supabase Auth

## Step 10: Monitor and Optimize

1. Go to Database → Logs to monitor query performance
2. Use the Dashboard to view table sizes and query statistics
3. Set up alerts for database usage

## Migration Benefits

### What You Gain:
- **Managed Database**: No need to manage PostgreSQL yourself
- **Real-time Subscriptions**: Built-in real-time capabilities
- **Vector Search**: Native support for AI embeddings
- **Authentication**: Built-in user management
- **Dashboard**: Web interface for database management
- **Backups**: Automatic backups and point-in-time recovery
- **Scaling**: Automatic scaling based on usage

### What Stays the Same:
- **SQLAlchemy Models**: Your existing models work unchanged
- **API Endpoints**: No changes needed to your FastAPI routes
- **Business Logic**: All your application logic remains the same

## Troubleshooting

### Common Issues:

1. **Connection Refused**: Check your DATABASE_URL format
2. **Permission Denied**: Ensure you're using the correct API keys
3. **Migration Errors**: Run migrations in the correct order
4. **Vector Search Not Working**: Ensure the `vector` extension is enabled

### Getting Help:
- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Discord](https://discord.supabase.com)
- [GitHub Issues](https://github.com/supabase/supabase/issues)

## Next Steps

1. **Test thoroughly** in a development environment first
2. **Migrate data** from your existing database if needed
3. **Update deployment** scripts to use new environment variables
4. **Monitor performance** and optimize queries as needed
5. **Explore Supabase features** like real-time subscriptions and edge functions

## Cost Considerations

Supabase offers a generous free tier:
- 500MB database size
- 2GB bandwidth
- 50,000 monthly active users
- 2GB file storage

For production use, consider the Pro plan ($25/month) which includes:
- 8GB database size
- 250GB bandwidth
- 100,000 monthly active users
- 100GB file storage
- Point-in-time recovery
