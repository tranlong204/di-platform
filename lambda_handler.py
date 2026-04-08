"""
AWS Lambda entrypoint for FastAPI application.
"""

from mangum import Mangum

from main import app

handler = Mangum(app, lifespan="off")
