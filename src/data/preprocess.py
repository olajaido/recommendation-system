import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

# Add the project root to the Python path to import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_movielens_100k():
    """
    Load the MovieLens 100K dataset from the raw data directory.
    
    Returns:
        tuple: (ratings_df, movies_df, users_df)
    """
    config = load_config()
    movielens_config = config["movielens"]
    raw_dir = movielens_config["paths"]["raw"]
    dataset_path = Path(raw_dir) / f"ml-{movielens_config['dataset_size']}"
    
    logger.info(f"Loading MovieLens 100K dataset from {dataset_path}")
    
    # Load ratings
    ratings_file = dataset_path / "u.data"
    ratings_df = pd.read_csv(
        ratings_file, 
        sep="\t", 
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    
    # Load movies (encoding Latin-1 for special characters)
    movies_file = dataset_path / "u.item"
    genre_columns = [
        "unknown", "Action", "Adventure", "Animation", "Children", 
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
        "Sci-Fi", "Thriller", "War", "Western"
    ]
    movies_cols = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_columns
    movies_df = pd.read_csv(
        movies_file, 
        sep="|", 
        encoding="latin-1", 
        names=movies_cols
    )
    
    # Load users
    users_file = dataset_path / "u.user"
    users_df = pd.read_csv(
        users_file, 
        sep="|", 
        names=["user_id", "age", "gender", "occupation", "zip_code"]
    )
    
    logger.info(f"Loaded ratings: {ratings_df.shape[0]} rows, {ratings_df.shape[1]} columns")
    logger.info(f"Loaded movies: {movies_df.shape[0]} rows, {movies_df.shape[1]} columns")
    logger.info(f"Loaded users: {users_df.shape[0]} rows, {users_df.shape[1]} columns")
    
    return ratings_df, movies_df, users_df

def clean_data(ratings_df, movies_df, users_df):
    """
    Clean and preprocess the MovieLens 100K dataset.
    
    Args:
        ratings_df (pd.DataFrame): Ratings dataframe
        movies_df (pd.DataFrame): Movies dataframe
        users_df (pd.DataFrame): Users dataframe
        
    Returns:
        tuple: Cleaned (ratings_df, movies_df, users_df)
    """
    logger.info("Cleaning and preprocessing the data")
    
    # Create a copy to avoid modifying the original data
    ratings = ratings_df.copy()
    movies = movies_df.copy()
    users = users_df.copy()
    
    # Convert timestamp to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Extract year from movie title
    # Movie titles in the format: "Movie Name (YYYY)"
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$', expand=False)
    
    # Convert release_date to datetime
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    
    # Create a genre feature that combines all genres
    genre_columns = movies.columns[5:24]  # Adjust based on actual column indices
    
    # Create a list of genres for each movie
    movies['genres'] = movies[genre_columns].apply(
        lambda x: [col for col, val in zip(genre_columns, x) if val == 1], 
        axis=1
    )
    
    # Drop the individual genre columns
    movies = movies.drop(columns=genre_columns)
    
    # Convert categorical variables in users
    users['gender'] = users['gender'].map({'M': 0, 'F': 1})
    
    logger.info("Data cleaning completed")
    
    return ratings, movies, users

def split_data(ratings_df):
    """
    Split the ratings data into training and testing sets.
    
    Args:
        ratings_df (pd.DataFrame): Ratings dataframe
        
    Returns:
        tuple: (train_df, test_df)
    """
    config = load_config()
    movielens_config = config["movielens"]
    split_config = movielens_config["split"]
    
    test_size = split_config["test_size"]
    validation_size = split_config["validation_size"]
    random_state = split_config["random_state"]
    
    logger.info(f"Splitting data: {test_size*100}% test, {validation_size*100}% validation")
    
    # First split into train+validation and test
    train_val_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=ratings_df['rating']  # Stratify by rating to preserve distribution
    )
    
    # Then split training into train and validation
    val_size_adjusted = validation_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size_adjusted, 
        random_state=random_state,
        stratify=train_val_df['rating']  # Stratify by rating
    )
    
    logger.info(f"Train set: {train_df.shape[0]} rows")
    logger.info(f"Validation set: {val_df.shape[0]} rows")
    logger.info(f"Test set: {test_df.shape[0]} rows")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, movies_df, users_df):
    """
    Save the processed and split data to files.
    
    Args:
        train_df (pd.DataFrame): Training set
        val_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
        movies_df (pd.DataFrame): Processed movies dataframe
        users_df (pd.DataFrame): Processed users dataframe
    """
    config = load_config()
    movielens_config = config["movielens"]
    processed_dir = Path(movielens_config["paths"]["processed"])
    
    # Create directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(processed_dir / "train_ratings.csv", index=False)
    val_df.to_csv(processed_dir / "val_ratings.csv", index=False)
    test_df.to_csv(processed_dir / "test_ratings.csv", index=False)
    movies_df.to_csv(processed_dir / "movies.csv", index=False)
    users_df.to_csv(processed_dir / "users.csv", index=False)
    
    logger.info(f"Processed data saved to {processed_dir}")

def main():
    """Main data preprocessing pipeline."""
    logger.info("Starting data preprocessing")
    
    # Load raw data
    ratings_df, movies_df, users_df = load_movielens_100k()
    
    # Clean data
    ratings_clean, movies_clean, users_clean = clean_data(ratings_df, movies_df, users_df)
    
    # Split data
    train_df, val_df, test_df = split_data(ratings_clean)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, movies_clean, users_clean)
    
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()