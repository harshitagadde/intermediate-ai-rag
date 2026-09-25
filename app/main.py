import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag_engine import rag_service

app = FastAPI(
    title="Intermediate AI RAG API",
    description="Custom Domain Document Q&A API built with FastAPI and FAISS",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_contexts: list[str]

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Engine Running"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    os.makedirs("temp", exist_ok=True)
    temp_path = f"temp/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_processed = rag_service.ingest_pdf(temp_path)
        os.remove(temp_path)
        return {
            "message": "Ingestion successful",
            "filename": file.filename,
            "chunks_processed": chunks_processed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        result = rag_service.query(user_query=request.question, k=request.top_k)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            retrieved_contexts=result["retrieved_contexts"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
