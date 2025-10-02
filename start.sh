#!/bin/bash

# Document Intelligence Platform - Startup Script

echo "🚀 Starting Document Intelligence Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "📊 Starting PostgreSQL..."
    brew services start postgresql@14
    sleep 3
fi

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "🔴 Starting Redis..."
    brew services start redis
    sleep 2
fi

# Create necessary directories
mkdir -p data uploads logs

echo "✅ All services are ready!"
echo ""
echo "🌐 Application URLs:"
echo "   API: http://localhost:8000"
echo "   Web Interface: http://localhost:3000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📝 To start the application:"
echo "   source venv/bin/activate"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "🎉 Document Intelligence Platform is ready to use!"
