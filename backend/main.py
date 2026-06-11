from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
import logging

from database import Base, engine
from face_engine import face_engine
from routes import persons, enrollment, authentication

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create storage directories
    os.makedirs("storage/faces", exist_ok=True)
    
    logger.info("Initializing face recognition engine...")
    face_engine.initialize()
    logger.info("Sentinel backend ready.")
    
    yield
    
    # Shutdown
    logger.info("Sentinel shutting down.")

app = FastAPI(
    title="Project Sentinel",
    description="AI-Powered Face Authentication Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for face snapshots
os.makedirs("storage/faces", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# Include routers
app.include_router(persons.router)
app.include_router(enrollment.router)
app.include_router(authentication.router)

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "Project Sentinel",
        "face_engine_ready": face_engine._initialized
    }
