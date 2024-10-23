from fastapi import FastAPI
from model_wrapper import llm
from vector_store import index
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()
# Load query engine
db = index()
query_engine = db.as_query_engine()

class QueryRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]] = []

# Define the POST endpoint
@app.get('/')
async def test():
    return '<h1>This service is live!</h1>'

@app.post("/ask-question")
async def ask_question(request: QueryRequest):
    
    # query from db thu RAG process using the fine-tuned llm
    result = query_engine.query(request.question)
    response = {
        "answer": result,
        "chat_history": request.chat_history + [(request.question, result)]
    }
    # stream db
    # query_engine_stream = index.as_query_engine(streaming=True)
    # response_stream = query_engine_stream.query(request.question)
    # response_stream.print_response_stream()
    return response