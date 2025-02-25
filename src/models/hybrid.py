import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import RecommenderBase
from src.data.content_features import ContentFeatureProcessor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class HybridRecommender(RecommenderBase):
    """
    Hybrid recommender that combines collaborative filtering with content-based features.
    
    This model uses a weighted approach to combine predictions from different models.
    """
    
    def __init__(self, cf_model, cf_weight=0.7, content_weight=0.3):
        """
        Initialize hybrid recommender.
        
        Args:
            cf_model: Collaborative filtering model
            cf_weight (float): Weight for collaborative filtering predictions
            content_weight (float): Weight for content-based predictions
        """
        super().__init__(name="Hybrid-Recommender")
        self.cf_model = cf_model
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.content_processor = ContentFeatureProcessor()
        self.user_item_matrix = None
        self.movies_df = None
    
    def fit(self, train_data):
        """
        Train the hybrid model.
        
        Args:
            train_data (pd.DataFrame): Training data with columns 'user_id', 'movie_id', 'rating'
        """
        logger.info(f"Training {self.name} model")
        
        # Make sure CF model is trained
        if not self.cf_model.is_fitted:
            logger.info("Training collaborative filtering model")
            self.cf_model.fit(train_data)
        
        # Process content features
        logger.info("Processing content features")
        self.content_processor.process_features()
        
        # Load movies data
        self.movies_df = self.content_processor.movies_df
        
        # Create user-item matrix for user preferences
        logger.info("Creating user preferences matrix")
        self.user_item_matrix = train_data.pivot(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        self.is_fitted = True
        logger.info(f"{self.name} model training completed")
    
    def get_user_content_preferences(self, user_id):
        """
        Calculate user preferences for content features based on rating history.
        
        Args:
            user_id (int): User ID
            
        Returns:
            np.ndarray: Weighted content preference vector
        """
        if user_id not in self.user_item_matrix.index:
            return None
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Filter movies that the user has rated
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return None
        
        # Get content vectors for rated movies
        content_vectors = []
        weights = []
        
        for movie_id, rating in rated_movies.items():
            vector = self.content_processor.get_content_vector(movie_id)
            if vector is not None:
                # Use rating as weight
                content_vectors.append(vector)
                weights.append(rating)
        
        if not content_vectors:
            return None
        
        # Convert to numpy arrays
        content_vectors = np.array(content_vectors)
        weights = np.array(weights).reshape(-1, 1)
        
        # Calculate weighted average of content vectors
        weighted_vector = np.average(content_vectors, axis=0, weights=weights.flatten())
        
        return weighted_vector
    
    def predict_content_based(self, user_id, movie_id):
        """
        Make content-based prediction for a user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        # Get user's content preferences
        user_preferences = self.get_user_content_preferences(user_id)
        
        if user_preferences is None:
            # Fall back to average rating for this movie
            avg_rating = self.movies_df[self.movies_df['movie_id'] == movie_id]['rating'].mean() \
                if 'rating' in self.movies_df.columns else 3.0
            return avg_rating
        
        # Get movie's content vector
        movie_vector = self.content_processor.get_content_vector(movie_id)
        
        if movie_vector is None:
            # Fall back to average rating for this movie
            avg_rating = self.movies_df[self.movies_df['movie_id'] == movie_id]['rating'].mean() \
                if 'rating' in self.movies_df.columns else 3.0
            return avg_rating
        
        # Calculate similarity between user preferences and movie vector
        similarity = cosine_similarity(
            user_preferences.reshape(1, -1),
            movie_vector.reshape(1, -1)
        )[0][0]
        
        # Scale similarity to rating scale (1-5)
        # Assuming similarity ranges from -1 to 1, scale to 1-5
        scaled_rating = ((similarity + 1) / 2) * 4 + 1
        
        return scaled_rating
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user-movie pair using hybrid approach.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get collaborative filtering prediction
        cf_prediction = self.cf_model.predict(user_id, movie_id)
        
        # Get content-based prediction
        content_prediction = self.predict_content_based(user_id, movie_id)
        
        # Combine predictions using weights
        hybrid_prediction = (
            self.cf_weight * cf_prediction + 
            self.content_weight * content_prediction
        )
        
        # Clip to valid rating range
        hybrid_prediction = max(1.0, min(5.0, hybrid_prediction))
        
        return hybrid_prediction
    
    def explain_recommendation(self, user_id, movie_id):
        """
        Provide explanation for a recommendation.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            dict: Explanation with different factors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining recommendations")
        
        # Get movie information
        movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
        
        # Get similar movies based on content
        similar_movies = self.content_processor.get_similar_movies(movie_id, n=5)
        
        # Get user's highly rated movies (if available)
        user_movies = []
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            user_rated_movies = user_ratings[user_ratings > 3.5].index.tolist()
            user_movies = self.movies_df[self.movies_df['movie_id'].isin(user_rated_movies)]['title'].tolist()[:3]
        
        # Get predictions from different components
        cf_prediction = self.cf_model.predict(user_id, movie_id)
        content_prediction = self.predict_content_based(user_id, movie_id)
        hybrid_prediction = self.predict(user_id, movie_id)
        
        # Create explanation
        explanation = {
            "movie_title": movie_info['title'],
            "movie_genres": movie_info['genres'],
            "predicted_rating": hybrid_prediction,
            "cf_component": cf_prediction,
            "content_component": content_prediction,
            "similar_movies": similar_movies['title'].tolist() if not similar_movies.empty else [],
            "user_liked_movies": user_movies
        }
        
        return explanation