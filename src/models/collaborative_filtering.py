import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from collections import defaultdict

from src.models.base import RecommenderBase
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class UserBasedCF(RecommenderBase):
    """
    User-based Collaborative Filtering recommender.
    
    This model recommends items based on similarity between users.
    """
    
    def __init__(self, k=30):
        """
        Initialize the User-Based CF model.
        
        Args:
            k (int): Number of neighbors to consider
        """
        super().__init__(name="User-Based CF")
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_means = None
    
    def fit(self, train_data):
        """
        Train the User-Based CF model.
        
        Args:
            train_data (pd.DataFrame): Training data with columns 'user_id', 'movie_id', 'rating'
        """
        logger.info(f"Training {self.name} model")
        
        # Create user-item matrix (sparse matrix representation as a DataFrame would be better for production)
        logger.info("Creating user-item matrix")
        self.user_item_matrix = train_data.pivot(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate mean rating for each user
        logger.info("Calculating user means")
        self.user_means = train_data.groupby('user_id')['rating'].mean()
        
        # Calculate user similarity matrix
        logger.info("Calculating user similarity matrix")
        self.user_similarity = pd.DataFrame(
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Fill similarity matrix (this is a naive implementation for demonstration)
        # In production, use a more efficient approach with sparse matrices
        for u1 in self.user_item_matrix.index:
            for u2 in self.user_item_matrix.index:
                if u1 == u2:
                    self.user_similarity.loc[u1, u2] = 1.0
                elif self.user_similarity.loc[u2, u1] != self.user_similarity.loc[u2, u1]:  # Check if NaN
                    # Calculate similarity
                    vector1 = self.user_item_matrix.loc[u1].values
                    vector2 = self.user_item_matrix.loc[u2].values
                    
                    # Skip if one of the vectors is all zeros
                    if np.sum(vector1) == 0 or np.sum(vector2) == 0:
                        self.user_similarity.loc[u1, u2] = 0
                        self.user_similarity.loc[u2, u1] = 0
                        continue
                    
                    # Calculate cosine similarity
                    similarity = 1 - cosine(vector1, vector2)
                    
                    # Handle NaN (happens when a vector is all zeros)
                    if np.isnan(similarity):
                        similarity = 0
                    
                    self.user_similarity.loc[u1, u2] = similarity
                    self.user_similarity.loc[u2, u1] = similarity
                else:
                    self.user_similarity.loc[u1, u2] = self.user_similarity.loc[u2, u1]
        
        self.is_fitted = True
        logger.info(f"{self.name} model training completed")
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Check if user_id exists in the training data
        if user_id not in self.user_item_matrix.index:
            # Return the global mean rating if user is not in training data
            return self.user_means.mean()
        
        # Check if movie_id exists in the training data
        if movie_id not in self.user_item_matrix.columns:
            # Return the user's mean rating if movie is not in training data
            return self.user_means.get(user_id, self.user_means.mean())
        
        # If user has already rated this movie, return the actual rating
        if self.user_item_matrix.loc[user_id, movie_id] > 0:
            return self.user_item_matrix.loc[user_id, movie_id]
        
        # Get similar users to the target user
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)
        
        # Filter to k most similar users who have rated the target movie
        neighbors = []
        for neighbor in similar_users.index:
            if neighbor != user_id and self.user_item_matrix.loc[neighbor, movie_id] > 0:
                neighbors.append(neighbor)
            if len(neighbors) >= self.k:
                break
        
        # If no neighbors have rated the movie, return user's mean rating
        if not neighbors:
            return self.user_means.get(user_id, self.user_means.mean())
        
        # Calculate weighted average rating from neighbors
        numerator = 0
        denominator = 0
        
        for neighbor in neighbors:
            # Get the similarity between target user and neighbor
            similarity = self.user_similarity.loc[user_id, neighbor]
            
            # Get the neighbor's rating for the movie
            rating = self.user_item_matrix.loc[neighbor, movie_id]
            
            numerator += similarity * rating
            denominator += abs(similarity)
        
        # Calculate predicted rating
        if denominator == 0:
            return self.user_means.get(user_id, self.user_means.mean())
        
        predicted_rating = numerator / denominator
        
        # Clip the rating to be within valid range [1, 5]
        predicted_rating = max(1, min(5, predicted_rating))
        
        return predicted_rating