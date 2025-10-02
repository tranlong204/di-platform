#!/bin/bash

# Document Intelligence Platform - Run Application Script

echo "🚀 Starting Document Intelligence Platform Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if services are running
if ! pg_isready -q; then
    echo "❌ PostgreSQL is not running. Please run ./start.sh first to start services."
    exit 1
fi

if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please run ./start.sh first to start services."
    exit 1
fi

# Create necessary directories
mkdir -p data uploads logs

echo "✅ Starting application services..."

# Start FastAPI in background
echo "📱 Starting FastAPI server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 3

# Start web server in background
echo "🌐 Starting web server..."
cd web && python3 -m http.server 3000 &
WEB_PID=$!

# Go back to project root
cd ..

echo ""
echo "🎉 Document Intelligence Platform is now running!"
echo ""
echo "🌐 Application URLs:"
echo "   API: http://localhost:8000"
echo "   Web Interface: http://localhost:3000"
echo "   API Documentation: http://localhost:8000/docs"
echo ""
echo "📝 Process IDs:"
echo "   FastAPI: $API_PID"
echo "   Web Server: $WEB_PID"
echo ""
echo "🛑 To stop the application, press Ctrl+C or run ./stop.sh"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down application..."
    kill $API_PID 2>/dev/null || true
    kill $WEB_PID 2>/dev/null || true
    echo "✅ Application stopped."
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
