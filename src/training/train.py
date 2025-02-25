import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import yaml
import time

from src.utils.logger import setup_logger
from src.models.collaborative_filtering import UserBasedCF
from src.evaluation.evaluate import evaluate_rating_predictions

logger = setup_logger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data():
    """
    Load the processed data.
    
    Returns:
        tuple: (train_df, val_df, test_df, movies_df, users_df)
    """
    config = load_config()
    movielens_config = config["movielens"]
    processed_dir = Path(movielens_config["paths"]["processed"])
    
    logger.info(f"Loading processed data from {processed_dir}")
    
    train_df = pd.read_csv(processed_dir / "train_ratings.csv")
    val_df = pd.read_csv(processed_dir / "val_ratings.csv")
    test_df = pd.read_csv(processed_dir / "test_ratings.csv")
    movies_df = pd.read_csv(processed_dir / "movies.csv")
    users_df = pd.read_csv(processed_dir / "users.csv")
    
    logger.info(f"Loaded training set: {train_df.shape[0]} rows")
    logger.info(f"Loaded validation set: {val_df.shape[0]} rows")
    logger.info(f"Loaded test set: {test_df.shape[0]} rows")
    
    return train_df, val_df, test_df, movies_df, users_df

def save_model(model, filename):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        filename (str): Filename to save the model
    """
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / filename
    logger.info(f"Saving model to {model_path}")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def train_user_based_cf():
    """
    Train a User-Based Collaborative Filtering model.
    
    Returns:
        UserBasedCF: Trained model
    """
    train_df, val_df, test_df, movies_df, users_df = load_data()
    
    # Create and train the model
    model = UserBasedCF(k=30)
    
    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    val_metrics = evaluate_rating_predictions(model, val_df)
    
    # Save model
    save_model(model, "user_based_cf.pkl")
    
    return model, val_metrics

def main():
    """Main training pipeline."""
    logger.info("Starting model training")
    
    model, metrics = train_user_based_cf()
    
    logger.info(f"Model validation metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    logger.info("Training process completed")

if __name__ == "__main__":
    main()