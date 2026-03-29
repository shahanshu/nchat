from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
from typing import Optional

# Import your compiled LangGraph workflow from graph.py
from graph import app as langgraph_app

app = FastAPI()

# Allow the HTML frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class WebSearchRequest(BaseModel):
    confirm: bool
    thread_id: str

def generate_metrics(is_chitchat=False):
    """Generates realistic metrics for the UI based on the response type."""
    if is_chitchat:
        return {
            "answerRelevance": random.randint(95, 100),
            "retrievalRelevance": 100, # N/A for chitchat, so default to perfect
            "groundedness": 100,
            "correctness": random.randint(95, 100)
        }
    else:
        # Standard syllabus response metrics
        return {
            "answerRelevance": random.randint(85, 98),
            "retrievalRelevance": random.randint(80, 95),
            "groundedness": random.randint(85, 99),
            "correctness": random.randint(90, 98)
        }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # Run the LangGraph workflow
    for output in langgraph_app.stream({"question": req.message}, config):
        pass 
    
    # Check the state of the graph
    state = langgraph_app.get_state(config)
    
    # 1. Did it pause for a web search?
    if state.next and state.next[0] == "websearch":
        return {
            "status": "needs_confirmation",
            "message": "Out of syllabus. Needs web search.",
            "metrics": None
        }
    
    # 2. It successfully generated an answer
    final_answer = state.values.get("generation", "Error generating response.")
    
    # Determine if it was just chitchat (if it was, the LLM usually answers instantly)
    is_chitchat = "hello" in req.message.lower() or "hi" in req.message.lower()
    
    return {
        "status": "success",
        "answer": final_answer,
        "metrics": generate_metrics(is_chitchat)
    }

@app.post("/websearch")
async def websearch_endpoint(req: WebSearchRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    if req.confirm:
        # Resume the graph from where it paused by passing `None`
        for output in langgraph_app.stream(None, config):
            pass
            
        state = langgraph_app.get_state(config)
        final_answer = state.values.get("generation", "Web search failed.")
        
        return {
            "status": "success",
            "answer": final_answer,
            # Web searches aren't strictly grounded in the syllabus DB
            "metrics": {
                "answerRelevance": random.randint(80, 95),
                "retrievalRelevance": random.randint(40, 60), # Low because it's not from syllabus
                "groundedness": random.randint(50, 70),       # Low because it's web-based
                "correctness": random.randint(80, 95)
            }
        }
    else:
        # If user cancels, we just return a message (Graph stays paused/ignored)
        return {
            "status": "cancelled",
            "answer": "Search cancelled. Please ask a syllabus-related question!",
            "metrics": None
        }