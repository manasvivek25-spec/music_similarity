"""
FastAPI Application Entrypoint for Music Similarity Research Pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.analysis import router as analysis_router
from src.api.routes.candidates import router as candidates_router
from src.api.routes.similarity import router as similarity_router
from src.api.routes.stats import router as stats_router
from src.api.schemas import HealthResponse

app = FastAPI(
    title="Musical Similarity & AI Detection Research API",
    description=(
        "Backend API for Statistical Analysis of Similarity in Human- and AI-Generated Music. "
        "Provides key-invariant DTW similarity, Suggestor Box segment localization, "
        "empirical null-hypothesis testing, candidate mimic ranking, and audio analysis."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration (allows seamless integration with separate frontend)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all dev frontends (Vite, React, Next.js, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Subrouters registration
# ---------------------------------------------------------------------------
app.include_router(similarity_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(candidates_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        engine="Key-Invariant DTW & Statistical Analysis",
    )


from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/", tags=["Health"])
    async def root():
        return {
            "message": "Musical Similarity & AI Detection Research API is running.",
            "docs_url": "/docs",
            "health_url": "/api/health",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)
