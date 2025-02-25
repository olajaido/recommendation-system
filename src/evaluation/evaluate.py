import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.evaluation.metrics import rmse, mae

logger = setup_logger(__name__)

def evaluate_rating_predictions(model, test_data):
    """
    Evaluate a recommender model on rating prediction task.
    
    Args:
        model: Trained recommender model
        test_data (pd.DataFrame): Test dataset with columns 'user_id', 'movie_id', 'rating'
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {model.name} on rating prediction")
    
    # Get true ratings
    y_true = test_data['rating'].values
    
    # Get predictions
    logger.info("Generating predictions")
    predictions = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Predicting"):
        pred = model.predict(row['user_id'], row['movie_id'])
        predictions.append(pred)
    
    y_pred = np.array(predictions)
    
    # Calculate metrics
    rmse_score = rmse(y_true, y_pred)
    mae_score = mae(y_true, y_pred)
    
    logger.info(f"RMSE: {rmse_score:.4f}")
    logger.info(f"MAE: {mae_score:.4f}")
    
    return {
        'rmse': rmse_score,
        'mae': mae_score
    }