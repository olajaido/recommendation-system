import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

from src.models.base import RecommenderBase
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MatrixFactorization(RecommenderBase):
    """
    Matrix Factorization recommender using SVD.
    
    This model learns latent factors for users and items through matrix factorization.
    """
    
    def __init__(self, n_factors=100, normalize=True):
        """
        Initialize Matrix Factorization model.
        
        Args:
            n_factors (int): Number of latent factors
            normalize (bool): Whether to normalize ratings by user mean
        """
        super().__init__(name=f"SVD-MF-{n_factors}")
        self.n_factors = n_factors
        self.normalize = normalize
        self.model = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_ids = None
        self.movie_ids = None
        self.user_means = None
        self.global_mean = None
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, train_data):
        """
        Train the Matrix Factorization model.
        
        Args:
            train_data (pd.DataFrame): Training data with columns 'user_id', 'movie_id', 'rating'
        """
        logger.info(f"Training {self.name} model")
        
        # Store unique user and movie IDs for mapping
        self.user_ids = sorted(train_data['user_id'].unique())
        self.movie_ids = sorted(train_data['movie_id'].unique())
        
        # Create mapping dictionaries
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        
        # Calculate user means for normalization
        if self.normalize:
            logger.info("Calculating user means for normalization")
            self.user_means = train_data.groupby('user_id')['rating'].mean()
            self.global_mean = train_data['rating'].mean()
        
        # Create user-item matrix (sparse)
        logger.info("Creating sparse user-item matrix")
        rows = [self.user_id_map[user_id] for user_id in train_data['user_id']]
        cols = [self.movie_id_map[movie_id] for movie_id in train_data['movie_id']]
        
        # Normalize ratings if enabled
        if self.normalize:
            ratings = [rating - self.user_means.get(user_id, self.global_mean) 
                      for rating, user_id in zip(train_data['rating'], train_data['user_id'])]
        else:
            ratings = train_data['rating'].values
        
        # Create sparse matrix
        matrix = csr_matrix((ratings, (rows, cols)), 
                           shape=(len(self.user_ids), len(self.movie_ids)))
        
        # Fit SVD model
        logger.info(f"Fitting SVD with {self.n_factors} factors")
        self.model.fit(matrix)
        
        # Extract user and item factors
        self.item_factors = self.model.components_
        self.user_factors = self.model.transform(matrix)
        
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
        
        # Handle cold start for users not in training set
        if user_id not in self.user_id_map:
            # Return global mean for unseen users
            return self.global_mean if self.normalize else 3.0
        
        # Handle cold start for items not in training set
        if movie_id not in self.movie_id_map:
            # Return user's mean rating for unseen items
            if self.normalize:
                return self.user_means.get(user_id, self.global_mean)
            else:
                return 3.0  # Default middle rating
        
        # Get user and movie indices
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        # Make prediction using dot product of user and item factors
        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[:, movie_idx]
        
        prediction = np.dot(user_vec, item_vec)
        
        # Add back user mean if normalization was used
        if self.normalize:
            prediction += self.user_means.get(user_id, self.global_mean)
        
        # Clip prediction to valid rating range
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction