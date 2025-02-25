import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path
import yaml
import time
import json

# Add project root to Python path if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger
from src.models.collaborative_filtering import UserBasedCF
from src.models.matrix_factorization import MatrixFactorization
from src.models.neural_cf import NeuralCollaborativeFiltering
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

def save_metrics(metrics, filename):
    """
    Save evaluation metrics to disk.
    
    Args:
        metrics (dict): Dictionary of metrics
        filename (str): Filename to save the metrics
    """
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    metrics_path = metrics_dir / filename
    logger.info(f"Saving metrics to {metrics_path}")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def train_neural_cf():
    """
    Train a Neural Collaborative Filtering model.
    
    Returns:
        tuple: (NeuralCollaborativeFiltering, dict) Trained model and validation metrics
    """
    train_df, val_df, test_df, movies_df, users_df = load_data()
    
    # Limit data size for faster training if needed
    # Using a smaller batch for demonstration purposes
    # train_sample = train_df.sample(n=50000, random_state=42) if len(train_df) > 50000 else train_df
    
    # Create and train the model
    model = NeuralCollaborativeFiltering(
        embedding_dim=64,
        hidden_layers=[128, 64, 32],
        batch_size=256,
        lr=0.001,
        n_epochs=5  # Reduced for demonstration, use 15-20 for better results
    )
    
    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    val_metrics = evaluate_rating_predictions(model, val_df)
    
    # Save model and metrics
    save_model(model, "neural_cf.pkl")
    save_metrics(val_metrics, "neural_cf_metrics.json")
    
    return model, val_metrics

def evaluate_on_test(model, name):
    """
    Evaluate a model on the test set.
    
    Args:
        model: Trained model
        name (str): Model name
    
    Returns:
        dict: Test metrics
    """
    _, _, test_df, _, _ = load_data()
    
    logger.info(f"Evaluating {name} on test set")
    metrics = evaluate_rating_predictions(model, test_df)
    
    # Save test metrics
    metrics_filename = f"{name.lower().replace(' ', '_')}_test_metrics.json"
    save_metrics(metrics, metrics_filename)
    
    return metrics

def main():
    """Main training pipeline."""
    logger.info("Starting Neural CF model training")
    
    # Train Neural CF
    ncf_model, ncf_val_metrics = train_neural_cf()
    
    # Evaluate on test set
    ncf_test_metrics = evaluate_on_test(ncf_model, "Neural_CF")
    
    # Print metrics
    logger.info(f"Neural CF Validation: RMSE={ncf_val_metrics['rmse']:.4f}, MAE={ncf_val_metrics['mae']:.4f}")
    logger.info(f"Neural CF Test: RMSE={ncf_test_metrics['rmse']:.4f}, MAE={ncf_test_metrics['mae']:.4f}")
    
    logger.info("Neural CF training process completed")

if __name__ == "__main__":
    main()