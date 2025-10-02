from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Document Intelligence Platform - Supabase Version", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "supabase"}
