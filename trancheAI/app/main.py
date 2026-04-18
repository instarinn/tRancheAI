from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import json
from app.trancheai import ask_tranchiq_bot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TranchIQ Chatbot API")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your website domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    user_id: str
    builder_id: str
    role: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sql_query: str | None = None
    data: list | None = None

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TranchIQ Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Log incoming payload
    logger.info(f"Incoming chat request: {json.dumps(req.dict())}")
    
    try:
        result = ask_tranchiq_bot(
            user_question=req.question,
            user_context={
                "user_id": req.user_id,
                "builder_id": req.builder_id,
                "role": req.role,
            }
        )

        return {
            "answer": result.get("final_answer", ""),
            "sql_query": result.get("sql_query"),
            "data": result.get("sql_result", []),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))