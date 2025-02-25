import os
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from datetime import datetime 
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import json
from fastapi.middleware.cors import CORSMiddleware 
from functools import lru_cache

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger
from src.serving.recommender import load_model, load_movie_data

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enhanced Movie Recommendation API",
    description="API for movie recommendations with multiple models and explanation capabilities.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Allow the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# In-memory cache for recommendations
recommendation_cache = {}

# Load models at startup
models = {}

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Loading models")
    
    model_dir = Path("models")
    
    # Load all available models
    model_files = {
        "user_based_cf": "user_based_cf.pkl",
        "matrix_factorization": "matrix_factorization.pkl",
        "neural_cf": "neural_cf.pkl",
        "hybrid_recommender": "hybrid_recommender.pkl"
    }
    
    for model_name, model_file in model_files.items():
        try:
            models[model_name] = load_model(model_dir / model_file)
            logger.info(f"Loaded {model_name} model")
        except FileNotFoundError:
            logger.warning(f"{model_name} model not found")
    
    if not models:
        logger.error("No models found!")
    else:
        logger.info(f"Loaded {len(models)} models")
    
    # Load movies data
    global movies_df
    movies_df = load_movie_data()
    logger.info(f"Loaded {len(movies_df)} movies")

# Define request/response models
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10
    model_name: Optional[str] = "hybrid_recommender"
    diversity_level: Optional[float] = 0.5  # Higher values promote diversity

class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    year: Optional[str] = None
    genres: Optional[List[str]] = None
    predicted_score: float

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    model_used: str
    processing_time_ms: float

class ExplanationRequest(BaseModel):
    user_id: int
    movie_id: int
    model_name: Optional[str] = "hybrid_recommender"

class ExplanationResponse(BaseModel):
    user_id: int
    movie_id: int
    movie_title: str
    explanation: Dict[str, Any]
    model_used: str

@lru_cache(maxsize=100)
def get_movie_details(movie_id):
    """Get movie details from movies dataframe (cached)."""
    movie = movies_df[movies_df['movie_id'] == movie_id]
    if len(movie) == 0:
        return None
    return movie.iloc[0]

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Enhanced Movie Recommendation API is running. Access /docs for API documentation."}

@app.get("/models")
async def get_available_models():
    """Get available models."""
    return {
        "available_models": list(models.keys()),
        "default_model": "hybrid_recommender" if "hybrid_recommender" in models else list(models.keys())[0] if models else None
    }

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_movie_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a specific user.
    
    Args:
        request (RecommendationRequest): Request parameters
        
    Returns:
        RecommendationResponse: Recommendations
    """
    start_time = time.time()
    
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
    
    # Check cache for recommendations
    cache_key = f"{request.user_id}_{model_name}_{request.num_recommendations}_{request.diversity_level}"
    if cache_key in recommendation_cache:
        recommendations = recommendation_cache[cache_key]
        processing_time = time.time() - start_time
        logger.info(f"Cache hit for {cache_key}")
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=model_name,
            processing_time_ms=processing_time * 1000
        )
    
    try:
        # Generate base recommendations (more than requested to allow for diversity filtering)
        num_base_recs = min(request.num_recommendations * 3, 100)
        
        # Get user's rated movies to exclude from recommendations
        user_rated_movies = []
        if hasattr(model, 'user_item_matrix') and request.user_id in model.user_item_matrix.index:
            user_ratings = model.user_item_matrix.loc[request.user_id]
            user_rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        # Generate candidate movies (excluding already rated ones)
        candidate_movies = movies_df[~movies_df['movie_id'].isin(user_rated_movies)]
        
        # Generate recommendations
        recs_df = model.recommend_for_user(request.user_id, candidate_movies, n=num_base_recs)
        
        # Apply diversity filter if using hybrid model and diversity_level > 0
        if model_name == "hybrid_recommender" and request.diversity_level > 0 and hasattr(model, 'content_processor'):
            # Get base recommendations
            base_movies = recs_df['movie_id'].tolist()
            
            # Initialize diversified recommendations with the top movie
            diversified_movies = [base_movies[0]]
            remaining_candidates = base_movies[1:]
            
            # Add movies one by one, promoting diversity
            while len(diversified_movies) < request.num_recommendations and remaining_candidates:
                # For each candidate, calculate average similarity to already selected movies
                max_diversity_score = -1
                most_diverse_movie = None
                
                for candidate in remaining_candidates:
                    # Get movie index in content processor
                    try:
                        candidate_idx = model.content_processor.indices[candidate]
                        
                        # Calculate average similarity to already selected movies
                        similarities = []
                        for selected in diversified_movies:
                            selected_idx = model.content_processor.indices[selected]
                            similarity = model.content_processor.cosine_sim[candidate_idx, selected_idx]
                            similarities.append(similarity)
                        
                        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                        
                        # Diversity score combines predicted rating and dissimilarity
                        rating = float(recs_df[recs_df['movie_id'] == candidate]['predicted_rating'].iloc[0])
                        diversity_score = (1 - request.diversity_level) * rating - request.diversity_level * avg_similarity
                        
                        if diversity_score > max_diversity_score:
                            max_diversity_score = diversity_score
                            most_diverse_movie = candidate
                    except:
                        continue
                
                if most_diverse_movie:
                    diversified_movies.append(most_diverse_movie)
                    remaining_candidates.remove(most_diverse_movie)
                else:
                    break
            
            # Filter recommendations to only include diversified movies
            recs_df = recs_df[recs_df['movie_id'].isin(diversified_movies)]
        
        # Limit to requested number
        recs_df = recs_df.head(request.num_recommendations)
        
        # Convert to response format
        recommendations = []
        for _, row in recs_df.iterrows():
            movie_details = get_movie_details(row['movie_id'])
            
            # Convert year to string or None if not available
            year_str = str(movie_details['year']) if pd.notna(movie_details['year']) else None
            
            # Convert genres to list if it's a string representation of a list
            genres = movie_details['genres']
            if isinstance(genres, str) and genres.startswith('['):
                import ast
                genres = ast.literal_eval(genres)
            
            recommendations.append(MovieRecommendation(
                movie_id=int(row['movie_id']),
                title=movie_details['title'],
                year=year_str,
                genres=genres,
                predicted_score=float(row['predicted_rating'])
            ))
        
        # Cache results
        recommendation_cache[cache_key] = recommendations
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=model_name,
            processing_time_ms=processing_time * 1000
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
@app.get("/batch-recommendations/{user_id}")
async def get_batch_recommendations(user_id: int):
    """
    Get pre-generated batch recommendations for a user.
    
    Args:
        user_id (int): User ID
        
    Returns:
        dict: Batch recommendations
    """
    try:
        # Check if user_id is valid
        if user_id < 1 or user_id > 943:
            raise HTTPException(status_code=400, detail="Invalid user ID. Must be between 1 and 943.")
        
        # Find most recent batch recommendations
        batch_dir = Path("data/recommendations")
        
        if not batch_dir.exists():
            raise HTTPException(status_code=404, detail="No batch recommendations available")
        
        # Get most recent batch file
        batch_files = list(batch_dir.glob("batch_recommendations_*.csv"))
        
        if not batch_files:
            raise HTTPException(status_code=404, detail="No batch recommendations available")
        
        # Sort by modification time (most recent first)
        latest_batch = sorted(batch_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        
        # Load recommendations
        all_recs = pd.read_csv(latest_batch)
        
        # Filter for user
        user_recs = all_recs[all_recs['user_id'] == user_id]
        
        if user_recs.empty:
            raise HTTPException(status_code=404, detail=f"No batch recommendations found for user {user_id}")
        
        # Convert to response format
        recommendations = []
        for _, row in user_recs.iterrows():
            movie_details = get_movie_details(row['movie_id'])
            
            # Convert year to string or None if not available
            year_str = str(movie_details['year']) if pd.notna(movie_details['year']) else None
            
            # Convert genres to list if it's a string representation of a list
            genres = movie_details['genres']
            if isinstance(genres, str) and genres.startswith('['):
                import ast
                genres = ast.literal_eval(genres)
            
            recommendations.append({
                "movie_id": int(row['movie_id']),
                "title": movie_details['title'],
                "year": year_str,
                "genres": genres,
                "predicted_score": float(row['predicted_rating']),
                "generation_time": row.get('timestamp', datetime.now().isoformat())
            })
        
        # Get batch metadata
        batch_name = latest_batch.name
        batch_time = datetime.fromtimestamp(latest_batch.stat().st_mtime).isoformat()
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "batch_source": batch_name,
            "batch_generation_time": batch_time
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving batch recommendations: {str(e)}")
    
@app.post("/explain", response_model=ExplanationResponse)
async def explain_recommendation(request: ExplanationRequest):
    """
    Explain why a movie was recommended to a user.
    
    Args:
        request (ExplanationRequest): Request parameters
        
    Returns:
        ExplanationResponse: Explanation
    """
    # Check if model exists
    if request.model_name not in models:
        available_models = list(models.keys())
        if not available_models:
            raise HTTPException(status_code=500, detail="No models are available")
        
        # Fall back to hybrid model if available, otherwise first model
        if "hybrid_recommender" in available_models:
            model_name = "hybrid_recommender"
        else:
            model_name = available_models[0]
        logger.warning(f"Requested model '{request.model_name}' not found. Using '{model_name}' instead.")
    else:
        model_name = request.model_name
    
    # Get model
    model = models[model_name]
    
    # Check if explanation is supported
    if model_name != "hybrid_recommender" or not hasattr(model, 'explain_recommendation'):
        # Create a basic explanation for non-hybrid models
        try:
            # Get movie details
            movie_details = get_movie_details(request.movie_id)
            if movie_details is None:
                raise HTTPException(status_code=404, detail=f"Movie ID {request.movie_id} not found")
            
            # Get predicted rating
            predicted_rating = model.predict(request.user_id, request.movie_id)
            
            # Basic explanation
            explanation = {
                "predicted_rating": float(predicted_rating),
                "genres": movie_details['genres'],
                "model_type": model_name,
                "note": "Detailed explanation only available for hybrid model"
            }
            
            return ExplanationResponse(
                user_id=request.user_id,
                movie_id=request.movie_id,
                movie_title=movie_details['title'],
                explanation=explanation,
                model_used=model_name
            )
        except Exception as e:
            logger.error(f"Error generating basic explanation: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")
    
    try:
        # Get explanation from hybrid model
        explanation = model.explain_recommendation(request.user_id, request.movie_id)
        
        return ExplanationResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            movie_title=explanation["movie_title"],
            explanation=explanation,
            model_used=model_name
        )
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")

@app.get("/metrics")
async def get_model_metrics():
    """Get performance metrics for all models."""
    metrics_dir = Path("metrics")
    
    if not metrics_dir.exists():
        return {"metrics": "No metrics available"}
    
    metrics = {}
    
    # Read all metric files
    for metric_file in metrics_dir.glob("*.json"):
        try:
            with open(metric_file, 'r') as f:
                model_metrics = json.load(f)
            
            model_name = metric_file.stem.replace("_metrics", "").replace("_test", "")
            is_test = "_test" in metric_file.stem
            
            if model_name not in metrics:
                metrics[model_name] = {}
            
            metrics[model_name]["test" if is_test else "validation"] = model_metrics
        except Exception as e:
            logger.error(f"Error reading metrics file {metric_file}: {e}")
    
    return {"metrics": metrics}

# Run with: uvicorn src.serving.enhanced_api:app --reload
if __name__ == "__main__":
    import uvicorn
    import json
    uvicorn.run("src.serving.enhanced_api:app", host="0.0.0.0", port=8000, reload=True)