"""
AWS Lambda entrypoint for FastAPI application.
"""

from mangum import Mangum

from main import app

# Disable lifespan hooks for Lambda cold starts.
handler = Mangum(app, lifespan="off")
