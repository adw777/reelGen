from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import uvicorn
from typing import Dict, Optional
import logging
from main import ContentPipeline
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Pipeline API",
    description="API for processing content and generating videos",
    version="1.0.0"
)

# Store job status in memory (in production, use a proper database)
jobs: Dict[str, Dict] = {}

class URLInput(BaseModel):
    url: HttpUrl

class JobStatus(BaseModel):
    job_id: str
    status: str
    start_time: str
    completion_time: Optional[str] = None
    files: Optional[Dict] = None
    error: Optional[str] = None

async def process_content_task(url: str, job_id: str):
    """Background task to process content"""
    pipeline = ContentPipeline()
    
    try:
        result = pipeline.process_content(url)
        jobs[job_id].update({
            "status": result["status"],
            "files": result.get("files"),
            "error": result.get("error"),
            "completion_time": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error processing content: {e}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completion_time": datetime.now().isoformat()
        })

@app.post("/process", response_model=JobStatus)
async def create_processing_job(url_input: URLInput, background_tasks: BackgroundTasks):
    """
    Start a new content processing job
    """
    try:
        # Generate job ID using timestamp
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "files": None,
            "error": None
        }
        
        # Add task to background tasks
        background_tasks.add_task(process_content_task, str(url_input.url), job_id)
        
        return jobs[job_id]
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    """
    try:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return jobs[job_id]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Content Pipeline API")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Content Pipeline API")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 