import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import yaml
import time
import json

from src.utils.logger import setup_logger
from src.models.collaborative_filtering import UserBasedCF
from src.models.matrix_factorization import MatrixFactorization
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

def train_user_based_cf():
    """
    Train a User-Based Collaborative Filtering model.
    
    Returns:
        tuple: (UserBasedCF, dict) Trained model and validation metrics
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
    
    # Save model and metrics
    save_model(model, "user_based_cf.pkl")
    save_metrics(val_metrics, "user_based_cf_metrics.json")
    
    return model, val_metrics

def train_matrix_factorization():
    """
    Train a Matrix Factorization model.
    
    Returns:
        tuple: (MatrixFactorization, dict) Trained model and validation metrics
    """
    train_df, val_df, test_df, movies_df, users_df = load_data()
    
    # Create and train the model
    model = MatrixFactorization(n_factors=50, normalize=True)
    
    start_time = time.time()
    model.fit(train_df)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    val_metrics = evaluate_rating_predictions(model, val_df)
    
    # Save model and metrics
    save_model(model, "matrix_factorization.pkl")
    save_metrics(val_metrics, "matrix_factorization_metrics.json")
    
    return model, val_metrics

def evaluate_on_test(models):
    """
    Evaluate multiple models on the test set.
    
    Args:
        models (list): List of tuples (model, name)
    
    Returns:
        dict: Dictionary mapping model names to test metrics
    """
    _, _, test_df, _, _ = load_data()
    
    logger.info("Evaluating models on test set")
    
    test_metrics = {}
    for model, name in models:
        logger.info(f"Evaluating {name} on test set")
        metrics = evaluate_rating_predictions(model, test_df)
        test_metrics[name] = metrics
    
    # Save test metrics
    save_metrics(test_metrics, "test_metrics.json")
    
    return test_metrics

def main():
    """Main training pipeline."""
    logger.info("Starting model training")
    
    # Train User-Based CF
    logger.info("Training User-Based Collaborative Filtering model")
    ucf_model, ucf_metrics = train_user_based_cf()
    
    # Train Matrix Factorization
    logger.info("Training Matrix Factorization model")
    mf_model, mf_metrics = train_matrix_factorization()
    
    # Compare models on validation set
    logger.info("Comparing models on validation set")
    logger.info(f"User-Based CF: RMSE={ucf_metrics['rmse']:.4f}, MAE={ucf_metrics['mae']:.4f}")
    logger.info(f"Matrix Factorization: RMSE={mf_metrics['rmse']:.4f}, MAE={mf_metrics['mae']:.4f}")
    
    # Evaluate on test set
    models = [(ucf_model, "User-Based CF"), (mf_model, "Matrix Factorization")]
    test_metrics = evaluate_on_test(models)
    
    # Print test metrics
    logger.info("Test metrics:")
    for model_name, metrics in test_metrics.items():
        logger.info(f"{model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    # Determine best model
    best_model = min(test_metrics.items(), key=lambda x: x[1]['rmse'])[0]
    logger.info(f"Best model based on test RMSE: {best_model}")
    
    logger.info("Training process completed")

if __name__ == "__main__":
    main()