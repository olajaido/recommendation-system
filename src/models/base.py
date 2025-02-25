from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class RecommenderBase(ABC):
    """Base class for recommender models."""
    
    def __init__(self, name="BaseRecommender"):
        """
        Initialize the recommender model.
        
        Args:
            name (str): Name of the recommender model
        """
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data):
        """
        Train the recommender model.
        
        Args:
            train_data (pd.DataFrame): Training data with columns 'user_id', 'movie_id', 'rating'
        """
        pass
    
    @abstractmethod
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        pass
    
    def predict_batch(self, user_movie_pairs):
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs (pd.DataFrame): Dataframe with 'user_id' and 'movie_id' columns
            
        Returns:
            np.ndarray: Predicted ratings
        """
        predictions = []
        for _, row in user_movie_pairs.iterrows():
            predictions.append(self.predict(row['user_id'], row['movie_id']))
        return np.array(predictions)
    
    def recommend_for_user(self, user_id, movie_pool, n=10):
        """
        Recommend top-N movies for a given user.
        
        Args:
            user_id (int): User ID
            movie_pool (pd.DataFrame): Pool of candidate movies
            n (int): Number of recommendations to generate
            
        Returns:
            pd.DataFrame: Top-N movie recommendations with predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Create pairs of user_id and each movie_id in the pool
        user_movie_pairs = pd.DataFrame({
            'user_id': [user_id] * len(movie_pool),
            'movie_id': movie_pool['movie_id'].values
        })
        
        # Predict ratings for each pair
        predictions = self.predict_batch(user_movie_pairs)
        
        # Create a dataframe with predictions
        recommendations = pd.DataFrame({
            'user_id': user_movie_pairs['user_id'],
            'movie_id': user_movie_pairs['movie_id'],
            'predicted_rating': predictions
        })
        
        # Sort by predicted rating (descending) and get top N
        top_n_recs = recommendations.sort_values('predicted_rating', ascending=False).head(n)
        
        return top_n_recs