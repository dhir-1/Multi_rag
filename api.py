"""
FastAPI Server & Web UI Backend for Multi-Agent RAG.

Serves the interactive research testing frontend and exposes the /api/query endpoint
which executes the LangGraph multi-agent pipeline.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

from agents.graph import rag_agent_app
from agents.state import AgentState
from agents.contextualizer import contextualize_followup
from config import MAX_ITERATIONS

app = FastAPI(title="Multi-Agent Financial Form 10-K RAG Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None
    max_iterations: Optional[int] = MAX_ITERATIONS


@app.get("/")
def serve_frontend():
    """Serves the test frontend UI."""
    index_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(index_path)


@app.post("/api/query")
def execute_query(req: QueryRequest):
    """
    Executes a research query through the LangGraph Multi-Agent pipeline,
    automatically contextualizing follow-up questions using conversation history.
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chat_history = req.chat_history or []
    standalone_query, was_rewritten, rewriter_tokens = contextualize_followup(query_text, chat_history)

    initial_state: AgentState = {
        "query": standalone_query,
        "original_query": query_text,
        "standalone_query": standalone_query,
        "chat_history": chat_history,
        "iteration_count": 0,
        "max_iterations": req.max_iterations or MAX_ITERATIONS,
        "prompt_tokens": rewriter_tokens if was_rewritten else 0,
        "completion_tokens": 0,
        "total_tokens": rewriter_tokens if was_rewritten else 0,
    }

    try:
        final_state = rag_agent_app.invoke(initial_state)
        final_state["original_query"] = query_text
        final_state["standalone_query"] = standalone_query
        final_state["was_rewritten"] = was_rewritten
        if was_rewritten:
            final_state["rewriter_tokens"] = rewriter_tokens
        return final_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "multi-agent-rag"}


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Multi-Agent RAG Web UI starting at: http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run(app, host="127.0.0.1", port=8000)
