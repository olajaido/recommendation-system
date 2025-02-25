import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add the project root to the Python path to import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger
from src.serving.recommender import load_model, load_movie_data, get_recommendations

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="API for movie recommendations using collaborative filtering.",
    version="1.0.0"
)

# Load models at startup
models = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Loading models")
    
    model_dir = Path("models")
    
    if (model_dir / "matrix_factorization.pkl").exists():
        models["matrix_factorization"] = load_model(model_dir / "matrix_factorization.pkl")
        logger.info("Loaded Matrix Factorization model")
    
    if (model_dir / "user_based_cf.pkl").exists():
        models["user_based_cf"] = load_model(model_dir / "user_based_cf.pkl")
        logger.info("Loaded User-Based CF model")
    
    if not models:
        logger.error("No models found!")

# Define request/response models
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10
    model_name: Optional[str] = "matrix_factorization"

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    year: Optional[str] = None
    predicted_score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    model_used: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Movie Recommendation API is running. Access /docs for API documentation."}

@app.get("/models")
async def get_available_models():
    """Get available models."""
    return {"available_models": list(models.keys())}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_movie_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a specific user.
    
    Args:
        request (RecommendationRequest): Request parameters
        
    Returns:
        RecommendationResponse: Recommendations
    """
    # Check if model exists
    if request.model_name not in models:
        available_models = list(models.keys())
        if not available_models:
            raise HTTPException(status_code=500, detail="No models are available")
        
        # Fall back to the first available model
        model_name = available_models[0]
        logger.warning(f"Requested model '{request.model_name}' not found. Using '{model_name}' instead.")
    else:
        model_name = request.model_name
    
    # Get model
    model = models[model_name]
    
    # Check if user_id is valid
    if request.user_id < 1 or request.user_id > 943:
        raise HTTPException(status_code=400, detail="Invalid user ID. Must be between 1 and 943.")
    
    try:
        # Generate recommendations
        recs_df = get_recommendations(model, request.user_id, n=request.num_recommendations)
        
        # Convert to response format
        recommendations = []
        for _, row in recs_df.iterrows():
            # Convert year to string or None if it's not available
            year_str = str(row['year']) if pd.notna(row['year']) else None
            
            recommendations.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=row['title'],
                year=year_str,  # Now it's properly converted to string
                predicted_score=float(row['predicted_score'])
            ))
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=model_name
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Run with: uvicorn src.serving.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)