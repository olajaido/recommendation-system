import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
        
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true (array-like): True ratings
        y_pred (array-like): Predicted ratings
        
    Returns:
        float: MAE score
    """
    return mean_absolute_error(y_true, y_pred)

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Precision@k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (ground truth) item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@k score
    """
    if len(recommended_items) == 0:
        return 0.0
        
    recommended_k = recommended_items[:k]
    relevant_and_recommended = set(relevant_items) & set(recommended_k)
    
    return len(relevant_and_recommended) / min(k, len(recommended_k))

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate Recall@k.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of relevant (ground truth) item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@k score
    """
    if len(relevant_items) == 0:
        return 0.0
        
    recommended_k = recommended_items[:k]
    relevant_and_recommended = set(relevant_items) & set(recommended_k)
    
    return len(relevant_and_recommended) / len(relevant_items)