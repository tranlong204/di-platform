#!/bin/bash

# Document Intelligence Platform - Stop Script

echo "🛑 Stopping Document Intelligence Platform..."

# Stop FastAPI application
echo "📱 Stopping FastAPI application..."
pkill -f "uvicorn app.main:app" 2>/dev/null || echo "   FastAPI not running"

# Stop web server
echo "🌐 Stopping web server..."
pkill -f "python3 -m http.server 3000" 2>/dev/null || echo "   Web server not running"

# Stop monitoring services (if running)
echo "📊 Stopping monitoring services..."
pkill -f "prometheus" 2>/dev/null || echo "   Prometheus not running"
pkill -f "grafana" 2>/dev/null || echo "   Grafana not running"

# Stop Redis (optional - comment out if you want Redis to keep running)
echo "🔴 Stopping Redis..."
brew services stop redis 2>/dev/null || echo "   Redis not running via brew"

# Stop PostgreSQL (optional - comment out if you want PostgreSQL to keep running)
echo "🐘 Stopping PostgreSQL..."
brew services stop postgresql@14 2>/dev/null || echo "   PostgreSQL not running via brew"

# Clean up any remaining processes
echo "🧹 Cleaning up processes..."
pkill -f "di-platform" 2>/dev/null || true

echo ""
echo "✅ Document Intelligence Platform stopped successfully!"
echo ""
echo "📝 To start again, run:"
echo "   ./start.sh"
echo ""
echo "💡 Note: Redis and PostgreSQL services have been stopped."
echo "   If you want to keep them running, edit this script and comment out those lines."
