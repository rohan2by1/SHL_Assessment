from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from app.models import ChatRequest, ChatResponse
from app.catalog_loader import load_and_prepare
from app.retriever import HybridRetriever
from app.agent import create_agent, Agent
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retriever: HybridRetriever = None
agent: Agent = None

app = FastAPI(title="SHL Assessment Agent v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    global retriever, agent
    catalog_file = os.getenv("CATALOG_FILE", "data/shl_catalog.json")
    logger.info(f"Loading catalog from {catalog_file}")
    assessments = load_and_prepare(catalog_file)
    logger.info(f"Loaded {len(assessments)} assessments")
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = HybridRetriever(embed_model, reranker_model)
    retriever.build_index(assessments)
    agent = create_agent(retriever)
    logger.info("Agent ready")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        reply, recs, end = await agent.process_conversation(request.messages)
        # Ensure recommendations list
        if not isinstance(recs, list):
            recs = []
        # Validate test_type code length
        for r in recs:
            if not r.test_type or len(r.test_type) > 1:
                r.test_type = "K"  # fallback
        return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")