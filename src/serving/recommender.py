import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: The loaded model
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_movie_data():
    """
    Load movie metadata.
    
    Returns:
        pd.DataFrame: Movies dataframe with metadata
    """
    config = load_config()
    movielens_config = config["movielens"]
    processed_dir = Path(movielens_config["paths"]["processed"])
    
    movies_path = processed_dir / "movies.csv"
    logger.info(f"Loading movie data from {movies_path}")
    
    movies_df = pd.read_csv(movies_path)
    return movies_df

def get_recommendations(model, user_id, n=10):
    """
    Generate movie recommendations for a specific user.
    
    Args:
        model: Trained recommender model
        user_id (int): User ID to generate recommendations for
        n (int): Number of recommendations to generate
        
    Returns:
        pd.DataFrame: Top N recommended movies with details
    """
    logger.info(f"Generating recommendations for user {user_id}")
    
    # Load movie data
    movies_df = load_movie_data()
    
    # Get user's rated movies
    if hasattr(model, 'user_item_matrix') and user_id in model.user_item_matrix.index:
        user_ratings = model.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
    else:
        rated_movies = []
    
    # Filter out movies already rated by the user
    candidate_movies = movies_df[~movies_df['movie_id'].isin(rated_movies)]
    
    # Generate recommendations
    recommendations = model.recommend_for_user(user_id, candidate_movies, n=n)
    
    # Merge with movie details
    recs_with_details = recommendations.merge(
        movies_df[['movie_id', 'title', 'year', 'genres']], 
        on='movie_id'
    )
    
    # Rename columns for clarity
    recs_with_details = recs_with_details.rename(columns={'predicted_rating': 'predicted_score'})
    
    return recs_with_details[['movie_id', 'title', 'year', 'genres', 'predicted_score']]

def main():
    """Main recommendation script."""
    # Load the trained model
    model = load_model("models/user_based_cf.pkl")
    
    # Get user ID from input
    try:
        user_id = int(input("Enter a user ID (1-943): "))
        if user_id < 1 or user_id > 943:
            logger.error("Invalid user ID. Please enter a number between 1 and 943.")
            return
    except ValueError:
        logger.error("Invalid input. Please enter a valid user ID.")
        return
    
    # Generate recommendations
    recommendations = get_recommendations(model, user_id, n=10)
    
    # Display recommendations
    print(f"\nTop 10 Movie Recommendations for User {user_id}:")
    print("=" * 80)
    for i, row in recommendations.iterrows():
        print(f"{i+1}. {row['title']} ({row['year']}) - Score: {row['predicted_score']:.2f}")
    
    print("\nRecommendation generation completed.")

if __name__ == "__main__":
    main()