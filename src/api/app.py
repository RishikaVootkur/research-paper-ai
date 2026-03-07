"""
FastAPI Backend
---------------
REST API for the Research Paper Intelligence Platform.

Run: uvicorn src.api.app:app --reload --port 8000
Docs: http://localhost:8000/docs (auto-generated Swagger UI)
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ============================================================
# Pydantic models (request/response schemas)
# ============================================================
# These define the exact shape of data going in and out of the API.
# FastAPI validates requests automatically using these.

class QueryRequest(BaseModel):
    """Request body for asking a question."""
    question: str = Field(..., description="Your question about ML/AI papers")
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)

class QueryResponse(BaseModel):
    """Response from the query endpoint."""
    answer: str
    sources: list
    route: str
    question_type: str
    quality_score: int
    num_papers: int
    hyde_used: bool
    agent_trace: list
    time_seconds: float

class IngestRequest(BaseModel):
    """Request body for ingesting papers."""
    query: str = Field(default=None, description="Search query for ArXiv")
    paper_ids: list[str] = Field(default=None, description="Specific ArXiv paper IDs")
    max_papers: int = Field(default=5, ge=1, le=20)

class IngestResponse(BaseModel):
    """Response from the ingest endpoint."""
    processed: int
    skipped: int
    failed: int
    total_chunks: int

class RecommendRequest(BaseModel):
    """Request body for recommendations."""
    paper_id: str = Field(default=None, description="Paper ID to find similar papers for")
    text: str = Field(default=None, description="Text description to find relevant papers")
    top_k: int = Field(default=5, ge=1, le=20)

class ClassifyRequest(BaseModel):
    """Request body for classification."""
    text: str = Field(..., description="Paper title + abstract to classify")

class PaperInfo(BaseModel):
    """Info about a single paper."""
    paper_id: str
    title: str
    chunks: int
    topic: str = None


# ============================================================
# App initialization with lifespan
# ============================================================
# We load the heavy ML models once at startup, not per-request.
# This is stored in app.state so all endpoints can access it.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    print("Loading models...")

    from src.agents.graph import AgentOrchestrator
    from src.ingestion.pipeline import IngestionPipeline
    from src.ml.recommender import PaperRecommender

    app.state.orchestrator = AgentOrchestrator()
    app.state.pipeline = IngestionPipeline()

    # Load recommender
    app.state.recommender = PaperRecommender()

    # Load topic classifier if available
    model_path = "models/topic_classifier"
    if os.path.exists(model_path):
        from src.ml.topic_classifier import TopicClassifier
        app.state.classifier = TopicClassifier.load(model_path)
    else:
        app.state.classifier = None
        print("  Topic classifier not found, classification endpoint disabled")

    print("All models loaded. API ready!")
    yield
    print("Shutting down...")


# ============================================================
# Create FastAPI app
# ============================================================

app = FastAPI(
    title="Research Paper Intelligence Platform",
    description="AI-powered research paper analysis with RAG and multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Check if the API is running."""
    stats = app.state.orchestrator.retriever_agent.retriever.hybrid_retriever.vector_store.get_collection_stats()
    return {
        "status": "healthy",
        "papers": stats["papers"],
        "chunks": stats["total_chunks"],
        "classifier_loaded": app.state.classifier is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about ML/AI research papers.

    The system routes your question to the appropriate specialist agent,
    retrieves relevant paper chunks, and generates a cited answer.
    """
    start = time.time()

    try:
        result = app.state.orchestrator.run(request.question)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            route=result["route"],
            question_type=result["question_type"],
            quality_score=result["quality_score"],
            num_papers=result["num_papers"],
            hyde_used=result.get("hyde_used", False),
            agent_trace=result["agent_trace"],
            time_seconds=round(time.time() - start, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Ingest new papers into the database.

    Provide either a search query or specific ArXiv paper IDs.
    """
    pipeline = app.state.pipeline

    try:
        if request.paper_ids:
            summary = pipeline.ingest_by_ids(request.paper_ids)
        elif request.query:
            summary = pipeline.ingest_by_query(request.query, max_papers=request.max_papers)
        else:
            raise HTTPException(status_code=400, detail="Provide either 'query' or 'paper_ids'")

        return IngestResponse(
            processed=summary["processed"],
            skipped=summary["skipped"],
            failed=summary["failed"],
            total_chunks=summary["total_chunks"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers", response_model=list[PaperInfo])
async def list_papers():
    """List all ingested papers with their topic classifications."""
    pipeline = app.state.pipeline
    log = pipeline.log

    papers = []
    for paper_id, info in log.get("ingested_papers", {}).items():
        papers.append(PaperInfo(
            paper_id=paper_id,
            title=info.get("title", "Unknown"),
            chunks=info.get("chunks", 0),
            topic=None,  # Could load from paper_topics.json
        ))

    return papers


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get paper recommendations based on a paper ID or text description."""
    recommender = app.state.recommender

    try:
        if request.paper_id:
            results = recommender.recommend_by_id(request.paper_id, top_k=request.top_k)
        elif request.text:
            results = recommender.recommend_by_text(request.text, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail="Provide either 'paper_id' or 'text'")

        return {"recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify(request: ClassifyRequest):
    """Classify a paper's topic based on its title + abstract."""
    if app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Topic classifier not loaded")

    try:
        result = app.state.classifier.predict(request.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))