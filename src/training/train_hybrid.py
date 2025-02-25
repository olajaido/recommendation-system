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
from src.models.matrix_factorization import MatrixFactorization
from src.models.neural_cf import NeuralCollaborativeFiltering
from src.models.hybrid import HybridRecommender
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

def load_model(filename):
    """
    Load a trained model from disk.
    
    Args:
        filename (str): Filename of the saved model
        
    Returns:
        object: The loaded model
    """
    model_path = Path("models") / filename
    logger.info(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def train_hybrid_model():
    """
    Train a hybrid recommender model.
    
    Returns:
        tuple: (HybridRecommender, dict) Trained model and validation metrics
    """
    train_df, val_df, test_df, movies_df, users_df = load_data()
    
    # Load base CF model (neural CF or matrix factorization)
    try:
        cf_model = load_model("neural_cf.pkl")
        logger.info("Using Neural CF as base model for hybrid")
    except FileNotFoundError:
        try:
            cf_model = load_model("matrix_factorization.pkl")
            logger.info("Using Matrix Factorization as base model for hybrid")
        except FileNotFoundError:
            # Train a new matrix factorization model
            logger.info("Training new Matrix Factorization model as base for hybrid")
            cf_model = MatrixFactorization(n_factors=50, normalize=True)
            cf_model.fit(train_df)
            save_model(cf_model, "matrix_factorization.pkl")
    
    # Create and train hybrid model
    model = HybridRecommender(cf_model, cf_weight=0.7, content_weight=0.3)
    
    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    val_metrics = evaluate_rating_predictions(model, val_df)
    
    # Save model and metrics
    save_model(model, "hybrid_recommender.pkl")
    save_metrics(val_metrics, "hybrid_recommender_metrics.json")
    
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
    logger.info("Starting hybrid model training")
    
    # Train Hybrid model
    hybrid_model, hybrid_val_metrics = train_hybrid_model()
    
    # Evaluate on test set
    hybrid_test_metrics = evaluate_on_test(hybrid_model, "Hybrid")
    
    # Print metrics
    logger.info(f"Hybrid Validation: RMSE={hybrid_val_metrics['rmse']:.4f}, MAE={hybrid_val_metrics['mae']:.4f}")
    logger.info(f"Hybrid Test: RMSE={hybrid_test_metrics['rmse']:.4f}, MAE={hybrid_test_metrics['mae']:.4f}")
    
    logger.info("Hybrid model training process completed")

if __name__ == "__main__":
    main()